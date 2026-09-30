from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot as pl
from common import parse_years, require
from facts import emit

MIN_STOPS = 10
VARIANTS = ("all", "min_stops", "distance")


def variant(full: pd.DataFrame, name: str, min_stops: int) -> tuple[pd.DataFrame, np.ndarray]:
    if name == "min_stops":
        full = full[full["n_stops"] >= min_stops]
    w = full["dist_nm"].to_numpy(float) if name == "distance" else np.ones(len(full))
    return full, w


def wshare(req: np.ndarray, w: np.ndarray, R: float) -> float:
    tot = w.sum()
    return float(w[req <= R].sum() / tot) if len(req) and tot > 0 else np.nan


def boot(full: pd.DataFrame, w: np.ndarray, ranges, B=pl.PILOT_BOOT, seed=pl.PILOT_SEED) -> dict:
    full = full.assign(_w=w)
    groups = [(g["req_lb"].to_numpy(float), g["_w"].to_numpy(float))
              for _, g in full.groupby("imo", sort=True)]
    k = len(groups)
    if k < 2:
        return {R: (np.nan, np.nan) for R in ranges}
    rng = np.random.default_rng(seed)
    R = np.asarray(ranges, float)
    feas = np.empty((B, len(R)))
    for b in range(B):
        idx = rng.integers(0, k, k)
        x = np.concatenate([groups[i][0] for i in idx])
        ww = np.concatenate([groups[i][1] for i in idx])
        tot = ww.sum()
        feas[b] = ((x[:, None] <= R) * ww[:, None]).sum(axis=0) / tot if tot > 0 else np.nan
    lo, hi = np.nanpercentile(feas, [2.5, 97.5], axis=0)
    return {r: (float(a), float(c)) for r, a, c in zip(ranges, lo, hi)}


def breakdown(sy: pd.DataFrame, fleet: pd.DataFrame | None, sets, reg_keys, pool,
              min_stops=MIN_STOPS, ranges=pl.R_GRID, reg_range=pl.RANGE) -> pd.DataFrame:
    rows = []
    for key, n, label in sets:
        p_all = pl.full_years(pl.rows_of(sy, key, n))
        p_all = p_all[p_all["year"].isin(pool)]
        f_all = pl.full_years(pl.rows_of(fleet, key, n)) if fleet is not None else None
        for v in VARIANTS:
            p, w = variant(p_all, v, min_stops)
            req = p["req_lb"].to_numpy(float)
            ships = p["imo"].nunique()
            ci = boot(p, w, ranges)
            if f_all is not None:
                f, fw = variant(f_all, v, min_stops)
                freq = f["req_lb"].to_numpy(float)
            for R in ranges:
                s = wshare(req, w, R)
                wl, wh = pl.wilson(s, ships)
                bl, bh = ci[R]
                lo, hi = (min(bl, wl), max(bh, wh)) if np.isfinite(bl) else (np.nan, np.nan)
                fs = wshare(freq, fw, R) if f_all is not None else np.nan
                rows.append({"set_key": key, "label": label, "range_nm": R, "variant": v,
                             "registered": (key, n) in reg_keys and R == reg_range,
                             "pilot_share": s, "pilot_lo": lo, "pilot_hi": hi,
                             "pilot_ship_years": int(len(p)), "pilot_ships": int(ships),
                             "fleet_share": fs,
                             "fleet_ship_years": int(len(f)) if f_all is not None else 0,
                             "reading": pl.reading(s, lo, hi, fs)})
    return pd.DataFrame(rows)


def owner_table(sy: pd.DataFrame, ships: pd.DataFrame, sets, pool, R, min_stops=MIN_STOPS):
    owner = ships.set_index("imo")["owner"] if "owner" in ships else pd.Series(dtype=str)
    first = sy.groupby("imo")["year"].min()
    rows = []
    for key, n, label in sets:
        p = pl.full_years(pl.rows_of(sy, key, n))
        p = p[p["year"].isin(pool)].assign(
            owner=lambda d: d["imo"].map(owner).fillna("").replace("", "unknown"),
            feas=lambda d: d["req_lb"] <= R,
            short=lambda d: d["n_stops"] < min_stops,
            first_year=lambda d: d["year"] == d["imo"].map(first))
        tot = p["dist_nm"].sum()
        for o, g in p.groupby("owner"):
            rows.append({"set_key": key, "label": label, "range_nm": R, "owner": o,
                         "ship_years": len(g), "ships": g["imo"].nunique(),
                         "feasible": int(g["feas"].sum()),
                         "short_ship_years": int(g["short"].sum()),
                         "short_first_year": int((g["short"] & g["first_year"]).sum()),
                         "short_feasible": int((g["short"] & g["feas"]).sum()),
                         "short_feasible_first_year": int((g["short"] & g["feas"]
                                                           & g["first_year"]).sum()),
                         "long_ship_years": int((~g["short"]).sum()),
                         "long_feasible": int((~g["short"] & g["feas"]).sum()),
                         "median_stops": float(g["n_stops"].median()),
                         "median_req_lb": float(g["req_lb"].median()),
                         "dist_nm": float(g["dist_nm"].sum()),
                         "short_dist_nm": float(g.loc[g["short"], "dist_nm"].sum()),
                         "dist_share": float(g["dist_nm"].sum() / tot) if tot > 0 else np.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pilot_dir", type=Path)
    ap.add_argument("--main", type=Path, default=None)
    ap.add_argument("--ships", type=Path, default=pl.SHIPS)
    ap.add_argument("--min-stops", type=int, default=MIN_STOPS)
    ap.add_argument("--pool", default=pl.POOL)
    ap.add_argument("--main-years", default=pl.MAIN_YEARS)
    ap.add_argument("--train", type=int, default=pl.TRAIN)
    ap.add_argument("--train-old", type=int, default=pl.TRAIN_OLD)
    ap.add_argument("--range", type=int, default=pl.RANGE)
    ap.add_argument("--n", type=int, default=pl.N)
    a = ap.parse_args(argv)
    pilot = a.pilot_dir.expanduser().resolve()
    require(pilot, "ship_years.csv.gz", stage="evaluate.py on the pilot directory")
    res = pilot / "results"
    res.mkdir(exist_ok=True)
    pool = parse_years(a.pool)
    sets = pl.pilot_sets(a.train, a.train_old, a.range, a.n)
    reg = pl.registered(a.train, a.range, a.n)
    reg_keys = {(k, n) for k, n, _ in reg}

    sy = pd.read_csv(pilot / "ship_years.csv.gz", dtype={"imo": str})
    sy = sy[sy["req_lb"].notna()]
    fleet = None
    if a.main is not None:
        fleet = pl.read_main_ship_years(a.main.expanduser().resolve(), {k for k, _, _ in sets},
                                        parse_years(a.main_years), extra=("n_stops", "dist_nm"))
    tab = breakdown(sy, fleet, sets, reg_keys, pool, a.min_stops, reg_range=a.range)
    tab.to_csv(res / "pilot_breakdown.csv", index=False)
    ships = pl.read_ships(a.ships)
    own = owner_table(sy, ships, sets, pool, a.range, a.min_stops)
    own.to_csv(res / "pilot_breakdown_owner.csv", index=False)

    pd.set_option("display.width", 160)
    show = tab[tab["range_nm"] == a.range]
    print(f"R = {a.range} nm, pilot {a.pool} vs fleet {a.main_years}"
          + ("" if fleet is not None else " (no --main: pilot side only)"))
    print(show[["label", "variant", "pilot_share", "pilot_lo", "pilot_hi", "pilot_ship_years",
                "fleet_share", "fleet_ship_years", "reading"]].round(3).to_string(index=False))
    print(own[own["set_key"] == pl.YAP4].drop(columns=["set_key", "label", "range_nm"])
          .round(3).to_string(index=False))

    f: dict = {"min_stops": int(a.min_stops), "range_nm": int(a.range),
               "fleet_side": fleet is not None}
    for key, n, short in reg:
        for v in VARIANTS:
            r = tab[(tab["set_key"] == key) & (tab["range_nm"] == a.range) & (tab["variant"] == v)]
            if not len(r):
                continue
            r = r.iloc[0]
            for col, name in (("pilot_share", "pilot"), ("pilot_lo", "pilot_lo"),
                              ("pilot_hi", "pilot_hi"), ("fleet_share", "fleet")):
                if np.isfinite(r[col]):
                    f[f"{short}_{v}_{name}_pct"] = 100.0 * r[col]
            f[f"{short}_{v}_ship_years"] = int(r["pilot_ship_years"])
            if fleet is not None:
                f[f"{short}_{v}_reading"] = r["reading"]
        o = own[own["set_key"] == key]
        if len(o):
            f[f"{short}_feasible"] = int(o["feasible"].sum())
            f[f"{short}_short_feasible"] = int(o["short_feasible"].sum())
            f[f"{short}_short_feasible_first_year"] = int(o["short_feasible_first_year"].sum())
            f[f"{short}_long_feasible"] = int(o["long_feasible"].sum())
    o = own[own["set_key"] == pl.BASELINE_KEY]
    if len(o):
        f["ship_years"] = int(o["ship_years"].sum())
        f["short_ship_years"] = int(o["short_ship_years"].sum())
        f["short_first_year"] = int(o["short_first_year"].sum())
        if o["dist_nm"].sum() > 0:
            f["short_dist_share_pct"] = 100.0 * o["short_dist_nm"].sum() / o["dist_nm"].sum()
    emit(res, "pilot_breakdown", {k: v for k, v in f.items()
                                  if isinstance(v, (str, bool)) or np.isfinite(v)})
    print(f"\nwrote {res}/pilot_breakdown.csv, pilot_breakdown_owner.csv")


if __name__ == "__main__":
    main()
