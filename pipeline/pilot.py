from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalise_imo, parse_ints, parse_years, require, to_utc
from evaluate import BASELINE_KEY
from facts import emit
from select_ports import HUBS, SET_COLS, external_frame, read_hubs, set_key

HERE = Path(__file__).resolve().parent
SHIPS = HERE / "reference" / "methanol_ships_public.csv"

TRAIN_YEARS = "2019,2024"
PILOT_NMAX = 20
RANK_RANGES = (0, 7000)
RANK_GROUPS = ("all", "container")
POOL = "2024-2026"
MAIN_YEARS = "2024-2025"
TRAIN, TRAIN_OLD, RANGE, N = 2024, 2019, 7000, 20
R_GRID = (5000, 7000, 10000, 15000, 20000)
PILOT_BOOT, PILOT_SEED = 1000, 20260928
METHANOL = "external:methanol6|-1|0|all"
YAP4 = "external:yap4|-1|0|all"
MAIN_SHIP_YEARS = ("ship_years.csv.gz", "ship_years_bunker.csv.gz",
                   "ship_years_methanol.csv.gz")


def pilot_sets(train=TRAIN, old=TRAIN_OLD, R=RANGE, n=N) -> list[tuple[str, int, str]]:
    return [(METHANOL, -1, "methanol ports named by DNV (6)"),
            (YAP4, -1, "Yap hubs (4)"),
            ("external:yap8|-1|0|all", -1, "Yap hubs (8)"),
            ("external:bunker10|-1|0|all", -1, "top-10 bunker ports"),
            (set_key("volume", train, 0, "all"), n, f"top {n} by calls, {train}"),
            (set_key("greedy", train, R, "all"), n, f"{n} chosen for continuity, {train}"),
            (set_key("greedy", train, R, "container"), n,
             f"{n} chosen for container continuity, {train}"),
            (set_key("greedy", old, R, "all"), n, f"{n} chosen for continuity, {old}"),
            (BASELINE_KEY, -1, "all nodes")]


def registered(train=TRAIN, R=RANGE, n=N) -> list[tuple[str, int, str]]:
    return [(YAP4, -1, "yap4"), (set_key("greedy", train, R, "all"), n, "greedy")]


def read_ships(path: Path = SHIPS) -> pd.DataFrame:
    text = "\n".join(l for l in Path(path).read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#"))
    s = pd.read_csv(io.StringIO(text), dtype=str)
    s["imo"] = normalise_imo(s["imo"])
    for c in ("name", "class"):
        if c not in s.columns:
            s[c] = ""
    return s.dropna(subset=["imo"]).drop_duplicates("imo").reset_index(drop=True)


def read_ids(pull: Path | None) -> pd.DataFrame | None:
    if pull is None or not (Path(pull) / "vessel_ids.csv").exists():
        return None
    ids = pd.read_csv(Path(pull) / "vessel_ids.csv", dtype=str).fillna("")
    ids["imo"] = normalise_imo(ids["imo"])
    return ids.drop_duplicates("imo", keep="last")


def rows_of(sy: pd.DataFrame, key: str, n: int) -> pd.DataFrame:
    part = sy[sy["set_key"] == key]
    return part if n < 0 else part[part["n_ports"] == n]


def read_main_ship_years(main: Path, keys, years, group="container",
                         extra=()) -> pd.DataFrame:
    cols = ["set_key", "n_ports", "imo", "year", "vessel_group", "req_lb", "n_breaks",
            "touches", *extra]
    parts = []
    for name in MAIN_SHIP_YEARS:
        p = main / name
        if not p.exists():
            continue
        for ch in pd.read_csv(p, usecols=cols, dtype={"imo": str}, chunksize=2_000_000):
            ch = ch[ch["set_key"].isin(keys) & ch["year"].isin(years)
                    & (ch["vessel_group"] == group) & ch["req_lb"].notna()]
            if len(ch):
                parts.append(ch)
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["set_key", "n_ports", "imo", "year"]).reset_index(drop=True)


def boot(sy: pd.DataFrame, ranges=R_GRID, B=PILOT_BOOT, seed=PILOT_SEED) -> dict:
    groups = [g["req_lb"].to_numpy(float) for _, g in sy.groupby("imo", sort=True)]
    k = len(groups)
    nan = {"median": (np.nan, np.nan), "feasible": {R: (np.nan, np.nan) for R in ranges}}
    if k < 2:
        return nan
    rng = np.random.default_rng(seed)
    R = np.asarray(ranges, float)
    med = np.empty(B)
    feas = np.empty((B, len(R)))
    for b in range(B):
        x = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        med[b] = np.median(x)
        feas[b] = (x[:, None] <= R).mean(axis=0)
    mlo, mhi = np.percentile(med, [2.5, 97.5])
    flo, fhi = np.percentile(feas, [2.5, 97.5], axis=0)
    return {"median": (float(mlo), float(mhi)),
            "feasible": {r: (float(a), float(b)) for r, a, b in zip(ranges, flo, fhi)}}


def full_years(part: pd.DataFrame) -> pd.DataFrame:
    return part[part["n_breaks"] == 0]


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0 or not np.isfinite(p):
        return np.nan, np.nan
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def summarise(part: pd.DataFrame, ranges=R_GRID, with_ci=True) -> dict:
    full = full_years(part)
    req = full["req_lb"].to_numpy(float)
    ci = boot(full, ranges) if with_ci else boot(full.iloc[:0], ranges)
    return {"n_ship_years": int(len(part)), "n_ships": int(full["imo"].nunique()),
            "n_full": int(len(full)),
            "touch_share": float(full["touches"].astype(str).eq("True").mean())
            if len(full) else np.nan,
            "median_req_lb": float(np.median(req)) if len(req) else np.nan,
            "median_lo": ci["median"][0], "median_hi": ci["median"][1],
            "feasible": {R: share_ci(req, R, full["imo"].nunique(), ci["feasible"][R])
                         for R in ranges}}


def share_ci(req: np.ndarray, R: float, n_ships: int, boot_ci) -> tuple[float, float, float]:
    p = float((req <= R).mean()) if len(req) else np.nan
    wl, wh = wilson(p, n_ships)
    bl, bh = boot_ci
    if not np.isfinite(bl):
        return p, np.nan, np.nan
    return p, min(bl, wl), max(bh, wh)


def set_table(sy: pd.DataFrame, sets, pool, ranges=R_GRID) -> pd.DataFrame:
    rows = []
    for key, n, label in sets:
        part = rows_of(sy, key, n)
        periods = [("pool", part[part["year"].isin(pool)], True)]
        periods += [(str(y), g, False) for y, g in part.groupby("year")]
        for period, p, ci in periods:
            if p.empty:
                continue
            s = summarise(p, ranges, with_ci=ci)
            for R in ranges:
                v, lo, hi = s["feasible"][R]
                rows.append({"set_key": key, "label": label,
                             "n_ports": n if n > 0 else int(p["n_ports"].iloc[0]),
                             "period": period, "n_ship_years": s["n_ship_years"],
                             "n_ships": s["n_ships"], "n_full": s["n_full"],
                             "touch_share": s["touch_share"],
                             "median_req_lb": s["median_req_lb"],
                             "median_lo": s["median_lo"], "median_hi": s["median_hi"],
                             "range_nm": R, "feasible_share": v,
                             "feasible_lo": lo, "feasible_hi": hi})
    return pd.DataFrame(rows)


def reading(pilot, lo, hi, fleet) -> str:
    vals = [pilot, lo, hi, fleet]
    if any(v is None or not np.isfinite(v) for v in vals):
        return "not comparable"
    if lo <= fleet <= hi:
        return "consistent"
    return "pilot higher" if fleet < lo else "pilot lower"


def compare_table(tab: pd.DataFrame, fleet: pd.DataFrame, sets, reg_keys,
                  ranges=R_GRID) -> pd.DataFrame:
    rows = []
    pooled = tab[tab["period"] == "pool"]
    for key, n, label in sets:
        f = full_years(rows_of(fleet, key, n))
        freq = f["req_lb"].to_numpy(float)
        for R in ranges:
            p = pooled[(pooled["set_key"] == key) & (pooled["range_nm"] == R)]
            pv, lo, hi = (p[["feasible_share", "feasible_lo", "feasible_hi"]].iloc[0]
                          if len(p) else (np.nan, np.nan, np.nan))
            fv = float((freq <= R).mean()) if len(freq) else np.nan
            rows.append({"set_key": key, "label": label, "range_nm": R,
                         "registered": (key, n) in reg_keys and R == RANGE,
                         "pilot_share": pv, "pilot_lo": lo, "pilot_hi": hi,
                         "pilot_ship_years": int(p["n_full"].iloc[0]) if len(p) else 0,
                         "fleet_share": fv, "fleet_ship_years": int(len(freq)),
                         "pilot_median_req_lb": float(p["median_req_lb"].iloc[0])
                         if len(p) else np.nan,
                         "fleet_median_req_lb": float(np.median(freq)) if len(freq) else np.nan,
                         "reading": reading(pv, lo, hi, fv)})
    return pd.DataFrame(rows)


def ship_table(stops: pd.DataFrame, nodes: pd.DataFrame, ships: pd.DataFrame,
               ids: pd.DataFrame | None, meth_nodes: set, pool) -> pd.DataFrame:
    node_of = nodes.set_index("anchorage_id")["node_id"]
    lead = nodes.drop_duplicates("node_id").set_index("node_id")["node_name"]
    s = stops.assign(node=stops["anchorage_id"].map(node_of),
                     year=to_utc(stops["visit_start"]).dt.year)
    rows = []
    for imo, g in s.groupby("imo"):
        gp = g[g["year"].isin(pool)]
        top = g["node"].dropna().map(lead).value_counts().head(3)
        rows.append({"imo": imo, "n_stops": len(g),
                     "first_visit": g["visit_start"].min(), "last_visit": g["visit_start"].max(),
                     "stops_without_node": int(g["node"].isna().sum()),
                     "methanol6_stop_share": float(gp["node"].isin(meth_nodes).mean())
                     if len(gp) else np.nan,
                     "top_nodes": "; ".join(f"{k} ({v})" for k, v in top.items())})
    per = pd.DataFrame(rows, columns=["imo", "n_stops", "first_visit", "last_visit",
                                      "stops_without_node", "methanol6_stop_share",
                                      "top_nodes"])
    out = ships[["imo", "name", "class"]].merge(per, on="imo", how="outer")
    out["listed"] = out["imo"].isin(set(ships["imo"]))
    out["n_stops"] = out["n_stops"].fillna(0).astype(int)
    if ids is not None:
        i = ids.set_index("imo")
        out["gfw_vessel_id"] = out["imo"].map(i["vessel_id"]).fillna("")
        out["gfw_name"] = out["imo"].map(i["shipname"]).fillna("") if "shipname" in i else ""
        out["in_gfw"] = out["gfw_vessel_id"] != ""
        if "n_identities" in i:
            out["gfw_identities"] = out["imo"].map(i["n_identities"]).fillna("")
    return out


def cmd_sets(a) -> None:
    pilot, main = a.pilot_dir.expanduser().resolve(), a.main.expanduser().resolve()
    require(main, "nodes.csv", "port_sets.csv", stage="the main run (build_stops.py, select_ports.py)")
    require(pilot, "stops.csv.gz", stage="build_stops.py on the pilot pull")
    if pilot == main:
        sys.exit("pilot_dir must not be the main output directory")
    shutil.copyfile(main / "nodes.csv", pilot / "nodes.csv")
    nodes = pd.read_csv(pilot / "nodes.csv", dtype={"anchorage_id": str, "node_id": str})
    labels = np.array(sorted(nodes["node_id"].dropna().unique()), dtype=object)

    ext = external_frame(read_hubs(a.hubs), nodes, labels)
    ps = pd.read_csv(main / "port_sets.csv", dtype={"node_id": str})
    keep = (ps["method"].isin(["volume", "greedy"])
            & ps["train_year"].isin(parse_ints(a.train_years))
            & ps["sel_range_nm"].isin(RANK_RANGES) & ps["groups"].isin(RANK_GROUPS)
            & (ps["rank"] <= a.nmax))
    rk = ps.loc[keep].reindex(columns=SET_COLS)
    out = pd.concat([f for f in (ext, rk) if len(f)], ignore_index=True)
    out.to_csv(pilot / "port_sets.csv", index=False)

    ships = read_ships(a.ships)
    grp = ships["group"] if "group" in ships.columns else "container"
    pd.DataFrame({"imo": ships["imo"], "group": grp, "name": ships["name"],
                  "class": ships["class"]}).to_csv(pilot / "vessels.csv", index=False)

    st = pd.read_csv(pilot / "stops.csv.gz", dtype=str, usecols=["imo", "anchorage_id"])
    unknown = ~st["anchorage_id"].isin(set(nodes["anchorage_id"]))
    unlisted = sorted(set(st["imo"]) - set(ships["imo"]))
    print(f"port_sets.csv: {out['set_key'].nunique()} sets "
          f"({ext['set_key'].nunique() if len(ext) else 0} external, "
          f"{rk['set_key'].nunique()} main-run rankings)")
    print(f"{len(st):,} pilot stops, {st['imo'].nunique()} ships; "
          f"{100 * unknown.mean():.2f}% at anchorages the main run never saw (no node)")
    if unlisted:
        print(f"  WARNING: stops for ships not on the list: {', '.join(unlisted)}")
    emit(pilot, "pilot_sets", {
        "sets": int(out["set_key"].nunique()),
        "external_sets": int(ext["set_key"].nunique()) if len(ext) else 0,
        "rankings": int(rk["set_key"].nunique()),
        "ships_listed": int(len(ships)), "ships_with_stops": int(st["imo"].nunique()),
        "stops": int(len(st)),
        "stops_without_node_pct": 100.0 * float(unknown.mean()) if len(st) else 0.0,
        "train_years": a.train_years, "nmax": int(a.nmax)})


def cmd_analyze(a) -> None:
    pilot, main = a.pilot_dir.expanduser().resolve(), a.main.expanduser().resolve()
    require(pilot, "ship_years.csv.gz", "stops.csv.gz", "nodes.csv", "port_sets.csv",
            stage="evaluate.py on the pilot directory")
    res = pilot / "results"
    res.mkdir(exist_ok=True)
    pool, main_years = parse_years(a.pool), parse_years(a.main_years)
    sets = pilot_sets(a.train, a.train_old, a.range, a.n)
    reg = registered(a.train, a.range, a.n)
    reg_keys = {(k, n) for k, n, _ in reg}

    sy = pd.read_csv(pilot / "ship_years.csv.gz", dtype={"imo": str})
    sy = sy[sy["req_lb"].notna()]
    if a.group:
        sy = sy[sy["vessel_group"] == a.group]
        res = res / a.group
        res.mkdir(exist_ok=True)
    tab = set_table(sy, sets, pool)
    tab.to_csv(res / "pilot_sets.csv", index=False)

    fleet = read_main_ship_years(main, {k for k, _, _ in sets}, main_years,
                                 group=a.group or "container")
    cmp_ = compare_table(tab, fleet, sets, reg_keys)
    cmp_.to_csv(res / "pilot_compare.csv", index=False)

    nodes = pd.read_csv(pilot / "nodes.csv", dtype={"anchorage_id": str, "node_id": str})
    ps = pd.read_csv(pilot / "port_sets.csv", dtype={"node_id": str})
    meth_nodes = set(ps.loc[ps["set_key"] == METHANOL, "node_id"])
    stops = pd.read_csv(pilot / "stops.csv.gz", dtype=str,
                        usecols=["imo", "visit_start", "anchorage_id"])
    ships = read_ships(a.ships)
    ids = read_ids(a.pull)
    per = ship_table(stops, nodes, ships, ids, meth_nodes, pool)
    per.to_csv(res / "pilot_ships.csv", index=False)

    pd.set_option("display.width", 160)
    print(f"\n{len(ships)} ships listed, {int((per['n_stops'] > 0).sum())} with stops"
          + (f", {int(per['in_gfw'].sum())} found in GFW" if "in_gfw" in per else ""))
    show = tab[(tab["period"] == "pool") & (tab["range_nm"] == a.range)]
    print(f"\npooled {a.pool}, R = {a.range} nm (95% intervals resample ships)")
    print(show[["label", "n_ports", "n_ship_years", "n_full", "n_ships", "touch_share",
                "median_req_lb",
                "median_lo", "median_hi", "feasible_share", "feasible_lo",
                "feasible_hi"]].round(3).to_string(index=False))
    print(f"\npilot vs fleet container ship-years {a.main_years}, R = {a.range} nm")
    c = cmp_[cmp_["range_nm"] == a.range]
    print(c[["label", "pilot_share", "pilot_lo", "pilot_hi", "fleet_share",
             "fleet_ship_years", "reading"]].round(3).to_string(index=False))

    f: dict = {"ships_listed": int(len(ships)),
               "ships_with_stops": int((per["n_stops"] > 0).sum()),
               "pool": a.pool, "fleet_years": a.main_years, "range_nm": int(a.range),
               "n_ports": int(a.n)}
    if "in_gfw" in per:
        f["ships_in_gfw"] = int(per["in_gfw"].sum())
    base = tab[(tab["set_key"] == BASELINE_KEY) & (tab["period"] == "pool")]
    if len(base):
        f["ship_years_pool"] = int(base["n_ship_years"].iloc[0])
        f["ship_years_full_pool"] = int(base["n_full"].iloc[0])
        f["fully_observed_pct"] = 100.0 * base["n_full"].iloc[0] / base["n_ship_years"].iloc[0]
    for key, n, short in reg:
        r = cmp_[(cmp_["set_key"] == key) & (cmp_["range_nm"] == a.range)]
        if len(r):
            r = r.iloc[0]
            f[f"p1_{short}_reading"] = r["reading"]
            for col, name in (("pilot_share", "pilot"), ("pilot_lo", "pilot_lo"),
                              ("pilot_hi", "pilot_hi"), ("fleet_share", "fleet")):
                f[f"p1_{short}_{name}_pct"] = 100.0 * r[col]
    m = tab[(tab["set_key"] == METHANOL) & (tab["period"] == "pool")]
    if len(m):
        f["p2_methanol6_median_req_nm"] = m["median_req_lb"].iloc[0]
        f["p2_methanol6_median_lo_nm"] = m["median_lo"].iloc[0]
        f["p2_methanol6_median_hi_nm"] = m["median_hi"].iloc[0]
        for _, r in m.iterrows():
            f[f"p2_methanol6_feasible_{int(r['range_nm'])}_pct"] = 100.0 * r["feasible_share"]
        f["p3_methanol6_touch_pct"] = 100.0 * m["touch_share"].iloc[0]
    st = per.loc[per["n_stops"] > 0, "methanol6_stop_share"]
    if len(st.dropna()):
        f["p3_methanol6_stop_share_median_pct"] = 100.0 * float(st.median())
    mf = cmp_[(cmp_["set_key"] == METHANOL) & (cmp_["range_nm"] == a.range)]
    if len(mf):
        f["fleet_methanol6_feasible_pct"] = 100.0 * mf["fleet_share"].iloc[0]
    emit(res, "pilot", {k: v for k, v in f.items()
                        if isinstance(v, (str, bool)) or np.isfinite(v)})
    print(f"\nwrote {res}/pilot_ships.csv, pilot_sets.csv, pilot_compare.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sets", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("pilot_dir", type=Path)
        p.add_argument("--main", type=Path, required=True, help="the main run's output dir")
        p.add_argument("--ships", type=Path, default=SHIPS)
        if name == "sets":
            p.add_argument("--hubs", type=Path, default=HUBS)
            p.add_argument("--train-years", default=TRAIN_YEARS)
            p.add_argument("--nmax", type=int, default=PILOT_NMAX)
        else:
            p.add_argument("--pull", type=Path, default=None,
                           help="the pull directory, for vessel_ids.csv")
            p.add_argument("--pool", default=POOL)
            p.add_argument("--main-years", default=MAIN_YEARS)
            p.add_argument("--train", type=int, default=TRAIN)
            p.add_argument("--train-old", type=int, default=TRAIN_OLD)
            p.add_argument("--range", type=int, default=RANGE)
            p.add_argument("--n", type=int, default=N)
            p.add_argument("--group", default="",
                           help="only this vessel group (results/<group>/); default all "
                                "ship-years, compared with fleet container ships")
    a = ap.parse_args(argv)
    (cmd_sets if a.cmd == "sets" else cmd_analyze)(a)


if __name__ == "__main__":
    main()
