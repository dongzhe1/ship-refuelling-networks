from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emissions
from common import require, tagged, to_utc
from evaluate import BASELINE_KEY
from facts import emit

HERE = Path(__file__).resolve().parent
REGIONS = HERE / "reference" / "country_regions.csv"

TRAIN, EVAL, RANGE, N = 2019, 2024, 7000, 20
GROUPS = ["all", "container", "bulk", "tanker", "gas", "general", "roro"]
FUELS = ["hfo", "lng", "methanol", "ammonia", "lh2"]
JACCARD_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
FAIL_PP, CONTINUE_PP = 2.0, 5.0


def key_parts(key: str) -> tuple[str, int, int, str]:
    m, y, r, g = key.split("|")
    return m, int(y), int(r), g


def selected_key(method, year, R, groups):
    return f"{method}|{int(year)}|{int(R)}|{groups}"


def group_match(sel_groups: str, vessel_group: str) -> bool:
    return sel_groups == vessel_group or (sel_groups == "all" and vessel_group == "all")


def staleness(cov: pd.DataFrame) -> pd.DataFrame:
    c = cov[cov["method"].isin(["volume", "greedy"])].copy()
    c = c[[group_match(g, v) for g, v in zip(c["groups"], c["vessel_group"])]]
    keys = ["method", "sel_range_nm", "groups", "n_ports", "vessel_group", "range_nm"]
    cols = ["feasible_share", "feasible_share_w", "covered_share", "covered_share_w",
            "touch_share", "touch_share_w"]
    oracle = c[c["train_year"] == c["eval_year"]][keys + ["eval_year"] + cols]
    oracle = oracle.rename(columns={k: f"{k}_oracle" for k in cols})
    s = c.merge(oracle, on=keys + ["eval_year"], how="left")
    for k in cols:
        s[f"{k}_loss"] = s[f"{k}_oracle"] - s[k]
    s["lag"] = s["eval_year"] - s["train_year"]
    return s[keys + ["train_year", "eval_year", "lag"] + cols
             + [f"{k}_oracle" for k in cols] + [f"{k}_loss" for k in cols]]


def staleness_by_lag(st: pd.DataFrame, R=RANGE) -> pd.DataFrame:
    s = st[st["range_nm"] == st["sel_range_nm"].where(st["sel_range_nm"] > 0, R)]
    g = s.groupby(["method", "sel_range_nm", "groups", "n_ports", "vessel_group", "lag"])
    return g.agg(pairs=("feasible_share", "size"),
                 feasible=("feasible_share", "mean"),
                 feasible_oracle=("feasible_share_oracle", "mean"),
                 loss=("feasible_share_loss", "mean"),
                 loss_w=("feasible_share_w_loss", "mean"),
                 covered_loss=("covered_share_loss", "mean")).reset_index()


def transfer_terms(tr: pd.DataFrame) -> pd.DataFrame:
    t = tr.copy()
    f = lambda c: t[c].fillna(0.0)
    t["dF"] = t["F_y"] - t["F_t"]
    t["panel_term"] = (t["n_panel"] / t["n_y"]) * f("F_panel_y") \
        - (t["n_panel"] / t["n_t"]) * f("F_panel_t")
    t["entry_term"] = (t["n_entry"] / t["n_y"]) * f("F_entry")
    t["exit_term"] = -(t["n_exit"] / t["n_t"]) * f("F_exit")
    t["within_panel_change"] = t["F_panel_y"] - t["F_panel_t"]
    t["panel_share_of_n_y"] = t["n_panel"] / t["n_y"]
    return t


def criteria(tr: pd.DataFrame, train=TRAIN, eval_=EVAL, R=RANGE, n=N) -> pd.DataFrame:
    t = transfer_terms(tr)
    rows = []
    for method in ("greedy", "volume"):
        sel_r = R if method == "greedy" else 0
        for grp in GROUPS:
            sel_groups = "all"
            m = t[(t["method"] == method) & (t["train_year"] == train)
                  & (t["eval_year"] == eval_) & (t["sel_range_nm"] == sel_r)
                  & (t["n_ports"] == n) & (t["groups"] == sel_groups)
                  & (t["vessel_group"] == grp) & (t["range_nm"] == R)]
            if m.empty:
                continue
            r = m.iloc[0]
            drop_pp = -100 * r["dF"]
            panel_drop_pp = -100 * r["panel_share_of_n_y"] * r["within_panel_change"] \
                if not np.isnan(r["within_panel_change"]) else np.nan
            mostly_panel = bool(panel_drop_pp >= 0.5 * drop_pp) if drop_pp > 0 else False
            rows.append({"method": method, "vessel_group": grp, "train_year": train,
                         "eval_year": eval_, "range_nm": R, "n_ports": n,
                         "F_t": r["F_t"], "F_y": r["F_y"], "drop_pp": drop_pp,
                         "panel_drop_pp": panel_drop_pp,
                         "composition_pp": drop_pp - panel_drop_pp
                         if not np.isnan(panel_drop_pp) else np.nan,
                         "mostly_panel": mostly_panel,
                         "n_t": r["n_t"], "n_y": r["n_y"], "n_panel": r["n_panel"]})
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    c["verdict_group"] = np.where((c["drop_pp"] >= CONTINUE_PP) & c["mostly_panel"],
                                  "continues",
                                  np.where((c["drop_pp"] < FAIL_PP) | ~c["mostly_panel"],
                                           "fails", "inconclusive"))
    return c


def overall_verdict(c: pd.DataFrame) -> str:
    if c.empty:
        return "no data"
    g = c[(c["method"] == "greedy") & (c["vessel_group"] != "all")]
    if g.empty:
        g = c[c["method"] == "greedy"]
    if (g["verdict_group"] == "continues").any():
        return "continues"
    if (g["verdict_group"] == "fails").all():
        return "fails"
    return "inconclusive"


def vessel_endurance(vessels: pd.DataFrame) -> pd.DataFrame:
    v = vessels.copy()
    model = emissions.daily_fuel_at_service_t(v["main_kw"])
    rec = pd.to_numeric(v["consumption_tpd"], errors="coerce").to_numpy(float)
    both = (rec > 0) & (model > 0)
    ratio = pd.Series(rec[both] / model[both]).groupby(v["group"].to_numpy()[both]).median()
    glob = float(np.median(rec[both] / model[both])) if both.any() else 1.0
    scale = v["group"].map(ratio).fillna(glob).to_numpy(float)
    daily = np.where(rec > 0, rec, model * scale)
    v["daily_fuel_model_t"] = model
    v["consumption_scale"] = scale
    v["daily_fuel_used_t"] = daily
    v["endurance_source"] = np.where(rec > 0, "recorded", "model_scaled")
    for fuel in FUELS:
        v[f"endurance_{fuel}_nm"] = emissions.endurance_nm(
            v["fuel_capacity_m3"], v["service_kn"], None, daily, 1.0, fuel)
    v["endurance_hfo_recorded_nm"] = np.where(rec > 0, v["endurance_hfo_nm"], np.nan)
    return v


def endurance_summary(v: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in GROUPS:
        part = v if grp == "all" else v[v["group"] == grp]
        if part.empty:
            continue
        row = {"vessel_group": grp, "n_vessels": len(part),
               "share_with_capacity": float(part["fuel_capacity_m3"].notna().mean()),
               "share_with_endurance": float(part["endurance_hfo_nm"].notna().mean())}
        for fuel in FUELS:
            e = part[f"endurance_{fuel}_nm"].dropna()
            for q in (0.25, 0.5, 0.75):
                row[f"{fuel}_p{int(q*100)}"] = float(e.quantile(q)) if len(e) else np.nan
        both = part.dropna(subset=["consumption_tpd", "daily_fuel_model_t"])
        both = both[(both["consumption_tpd"] > 0) & (both["daily_fuel_model_t"] > 0)]
        row["n_recorded_consumption"] = len(both)
        row["model_over_recorded_median"] = float(
            (both["daily_fuel_model_t"] / both["consumption_tpd"]).median()) if len(both) else np.nan
        row["consumption_scale"] = float(part["consumption_scale"].median())
        rec = part[part["endurance_source"] == "recorded"]["endurance_hfo_nm"].dropna()
        row["hfo_p50_recorded_only"] = float(rec.median()) if len(rec) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def required_range(sy: pd.DataFrame) -> pd.DataFrame:
    full = sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]
    rows = []
    for (key, n, year), part in full.groupby(["set_key", "n_ports", "year"]):
        for grp in GROUPS:
            p = part if grp == "all" else part[part["vessel_group"] == grp]
            if p.empty:
                continue
            r = p["req_lb"]
            rows.append({"set_key": key, "n_ports": n, "year": year, "vessel_group": grp,
                         "n_full": len(p),
                         **{f"p{int(q*100)}": float(r.quantile(q)) for q in (.1, .25, .5, .75, .9)}})
    return pd.DataFrame(rows)


CDF_GRID = np.arange(0, 30001, 250)


def cdf_sets(train=TRAIN, eval_=EVAL, R=RANGE, n=N) -> list[tuple[str, int, int, str]]:
    return [(BASELINE_KEY, -1, eval_, "all nodes"),
            ("external:yap4|-1|0|all", -1, eval_, "Yap hubs (4)"),
            ("external:yap8|-1|0|all", -1, eval_, "Yap hubs (8)"),
            (selected_key("volume", eval_, 0, "all"), n, eval_, f"top {n} by calls"),
            (selected_key("greedy", eval_, R, "all"), n, eval_, f"{n} chosen for continuity"),
            (selected_key("greedy", train, R, "all"), n, eval_,
             f"{n} chosen for continuity on {train}"),
            (selected_key("greedy", train, R, "all"), n, train,
             f"{n} chosen for continuity on {train}")]


def required_range_cdf(sy: pd.DataFrame, train=TRAIN, eval_=EVAL, R=RANGE, n=N) -> pd.DataFrame:
    full = sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]
    rows = []
    for key, nn, year, label in cdf_sets(train, eval_, R, n):
        part = full[(full["set_key"] == key) & (full["year"] == year)]
        if nn > 0:
            part = part[part["n_ports"] == nn]
        for grp in GROUPS:
            p = part if grp == "all" else part[part["vessel_group"] == grp]
            if p.empty:
                continue
            v = np.sort(p["req_lb"].to_numpy(float))
            share = np.searchsorted(v, CDF_GRID, side="right") / len(v)
            rows.append(pd.DataFrame({"set_key": key, "label": label, "year": year,
                                      "vessel_group": grp, "n": len(v),
                                      "range_nm": CDF_GRID, "share_le": share}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["set_key", "label", "year", "vessel_group", "n", "range_nm", "share_le"])


def own_endurance(sy: pd.DataFrame, v: pd.DataFrame, weighted: bool = True) -> pd.DataFrame:
    full = sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]
    e = v.set_index("imo")[[f"endurance_{f}_nm" for f in FUELS]]
    j = full.join(e, on="imo", how="inner").dropna(subset=["endurance_hfo_nm"])
    rows = []
    for (key, n, year), part in j.groupby(["set_key", "n_ports", "year"]):
        for grp in GROUPS:
            p = part if grp == "all" else part[part["vessel_group"] == grp]
            if p.empty:
                continue
            row = {"set_key": key, "n_ports": n, "year": year, "vessel_group": grp,
                   "n": len(p)}
            w = p["w_total"].to_numpy(float) if weighted and "w_total" in p else None
            for f in FUELS:
                ok = (p["req_lb"] <= p[f"endurance_{f}_nm"]).to_numpy()
                row[f"fits_{f}"] = float(ok.mean())
                if w is not None and w.sum() > 0:
                    row[f"fits_{f}_w"] = float(w[ok].sum() / w.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def redeploy_loss(sy: pd.DataFrame, red: pd.DataFrame, train=TRAIN, R=RANGE, n=N) -> pd.DataFrame:
    key = selected_key("greedy", train, R, "all")
    s = sy[(sy["set_key"] == key) & (sy["n_ports"] == n) & (sy["n_breaks"] == 0)
           & sy["req_lb"].notna()]
    if s.empty:
        return pd.DataFrame()
    at = s[s["year"] == train].set_index("imo")[["req_lb", "vessel_group"]]
    rows = []
    for y in sorted(set(s["year"]) - {train}):
        ay = s[s["year"] == y].set_index("imo")["req_lb"].rename("req_y")
        j = at.join(ay, how="inner")
        rj = red[(red["train_year"] == train) & (red["year"] == y)].set_index("imo")["jaccard"]
        j = j.join(rj, how="inner")
        if j.empty:
            continue
        j["bin"] = pd.cut(j["jaccard"], JACCARD_BINS, right=False, include_lowest=True)
        for grp in GROUPS:
            p = j if grp == "all" else j[j["vessel_group"] == grp]
            for b, q in p.groupby("bin", observed=True):
                feas_t = q["req_lb"] <= R
                lost = feas_t & (q["req_y"] > R)
                gained = ~feas_t & (q["req_y"] <= R)
                rows.append({"train_year": train, "eval_year": y, "vessel_group": grp,
                             "jaccard_bin": str(b), "jaccard_lo": b.left, "n": len(q),
                             "n_feasible_t": int(feas_t.sum()),
                             "lost_share": float(lost.sum() / feas_t.sum()) if feas_t.sum() else np.nan,
                             "gained_share": float(gained.sum() / (~feas_t).sum()) if (~feas_t).sum() else np.nan,
                             "median_req_ratio": float((q["req_y"] / q["req_lb"]).median())})
    return pd.DataFrame(rows)


def ship_year_regions(stops: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    s = stops.assign(year=to_utc(stops["visit_start"]).dt.year)
    s = s.merge(regions[["iso3", "region", "subregion"]], on="iso3", how="left")
    s["subregion"] = s["subregion"].fillna("Unknown")
    g = s.groupby(["imo", "year", "subregion"]).size().rename("k").reset_index()
    g = g.sort_values(["imo", "year", "k", "subregion"], ascending=[True, True, False, True])
    top = g.drop_duplicates(["imo", "year"]).rename(columns={"subregion": "home_subregion"})
    tot = g.groupby(["imo", "year"])["k"].sum().rename("n")
    top = top.join(tot, on=["imo", "year"])
    top["home_share"] = top["k"] / top["n"]
    return top[["imo", "year", "home_subregion", "home_share"]]


def region_loss(sy: pd.DataFrame, reg: pd.DataFrame, train=TRAIN, eval_=EVAL, R=RANGE,
                n=N) -> pd.DataFrame:
    key = selected_key("greedy", train, R, "all")
    s = sy[(sy["set_key"] == key) & (sy["n_ports"] == n) & (sy["n_breaks"] == 0)
           & sy["req_lb"].notna()]
    at = s[s["year"] == train].set_index("imo")["req_lb"]
    ay = s[s["year"] == eval_].set_index("imo")["req_lb"].rename("req_y")
    j = at.to_frame("req_t").join(ay, how="inner")
    home = reg[reg["year"] == train].set_index("imo")[["home_subregion"]]
    j = j.join(home, how="left").fillna({"home_subregion": "Unknown"})
    same = reg[reg["year"] == eval_].set_index("imo")["home_subregion"].rename("home_y")
    j = j.join(same, how="left")
    rows = []
    for r, q in j.groupby("home_subregion"):
        rows.append({"home_subregion": r, "n_panel": len(q),
                     "F_t": float((q["req_t"] <= R).mean()),
                     "F_y": float((q["req_y"] <= R).mean()),
                     "moved_share": float((q["home_y"] != r).mean())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["drop_pp"] = 100 * (out["F_t"] - out["F_y"])
    return out.sort_values("n_panel", ascending=False) if not out.empty else out


def yap_comparison(cov: pd.DataFrame) -> pd.DataFrame:
    keep = cov["method"].str.startswith("external") | (cov["set_key"] == BASELINE_KEY) \
        | ((cov["method"].isin(["volume", "greedy"])) & cov["n_ports"].isin([5, 10])
           & (cov["groups"] == "all"))
    cols = ["set_key", "method", "train_year", "sel_range_nm", "n_ports", "eval_year",
            "vessel_group", "range_nm", "n_full", "touch_share", "touch_share_w",
            "feasible_share", "feasible_share_w", "covered_share", "covered_share_w"]
    return cov.loc[keep, cols].reset_index(drop=True)


def backup_table(cov: pd.DataFrame) -> pd.DataFrame:
    b = cov[cov["method"].astype(str).str.startswith("backup")]
    if b.empty:
        return b
    ey = int(b["train_year"].iloc[0])
    R = int(b["sel_range_nm"].iloc[0])
    oracle = cov[(cov["method"] == "greedy") & (cov["train_year"] == ey)
                 & (cov["sel_range_nm"] == R) & (cov["groups"] == "all")]
    t = pd.concat([b.assign(kind="backup"), oracle.assign(kind="oracle")])
    t = t[(t["eval_year"] == ey)]
    return t[["kind", "set_key", "n_ports", "eval_year", "vessel_group", "range_nm",
              "feasible_share", "feasible_share_w", "covered_share", "covered_share_w"]]


def volume_vs_greedy(cov: pd.DataFrame, R=RANGE) -> pd.DataFrame:
    c = cov[cov["method"].isin(["volume", "greedy"]) & (cov["groups"] == "all")
            & (cov["range_nm"] == R) & ((cov["method"] == "volume") | (cov["sel_range_nm"] == R))]
    idx = ["train_year", "eval_year", "n_ports", "vessel_group"]
    p = c.pivot_table(index=idx, columns="method",
                      values=["feasible_share", "covered_share", "feasible_share_w"])
    p.columns = [f"{a}_{b}" for a, b in p.columns]
    p = p.reset_index()
    if {"feasible_share_greedy", "feasible_share_volume"} <= set(p.columns):
        p["feasible_gap"] = p["feasible_share_greedy"] - p["feasible_share_volume"]
    p["in_sample"] = p["train_year"] == p["eval_year"]
    return p


def sensitivity(out_dir: Path, main: pd.DataFrame, train=TRAIN, eval_=EVAL, R=RANGE,
                n=N) -> pd.DataFrame:
    rows = []
    cells = [("greedy", train, train), ("greedy", train, eval_), ("greedy", eval_, eval_),
             ("volume", train, train), ("volume", train, eval_)]
    runs = [("main", main)] + [(p.stem.split("coverage_", 1)[1], pd.read_csv(p))
                               for p in sorted(out_dir.glob("coverage_*.csv"))]
    for tag, cov in runs:
        c = cov[(cov["vessel_group"] == "all") & (cov["range_nm"] == R)]
        for method, t, y in cells:
            sel_r = R if method == "greedy" else 0
            m = c[(c["method"] == method) & (c["train_year"] == t) & (c["eval_year"] == y)
                  & (c["sel_range_nm"] == sel_r) & (c["n_ports"] == n) & (c["groups"] == "all")]
            if len(m):
                rows.append({"run": tag, "cell": f"{method} {t}->{y}",
                             "feasible_share": float(m["feasible_share"].iloc[0])})
        b = c[(c["set_key"] == BASELINE_KEY) & (c["eval_year"] == eval_)]
        if len(b):
            rows.append({"run": tag, "cell": f"all-nodes {eval_}",
                         "feasible_share": float(b["feasible_share"].iloc[0])})
    s = pd.DataFrame(rows)
    return s.pivot_table(index="cell", columns="run", values="feasible_share").reset_index() \
        if not s.empty else s


def manifest(results: Path) -> None:
    lines = []
    for p in sorted(results.iterdir()):
        if p.name == "MANIFEST.txt" or not p.is_file():
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        rows = ""
        if p.suffix == ".csv":
            with open(p) as fh:
                rows = sum(1 for _ in fh) - 1
        lines.append(f"{p.name:<40} {str(rows):>9} {h}")
    (results / "MANIFEST.txt").write_text("file rows sha256[:16]\n" + "\n".join(lines) + "\n")


def headline_facts(cov, st, crit, yap, own, endur, TRAIN=TRAIN, EVAL=EVAL, RANGE=RANGE,
                   N=N) -> dict:
    f = {"train_year": TRAIN, "eval_year": EVAL, "range_nm": RANGE, "n_ports": N,
         "verdict": overall_verdict(crit)}
    c = cov[(cov["range_nm"] == RANGE)]
    b = c[(c["set_key"] == BASELINE_KEY) & (c["vessel_group"] == "all") & (c["eval_year"] == EVAL)]
    if len(b):
        f["baseline_feasible_pct"] = 100 * float(b["feasible_share"].iloc[0])
        f["baseline_median_req_nm"] = float(b["median_req_lb"].iloc[0])

    def cell(method, t, y, grp="all"):
        sel_r = RANGE if method == "greedy" else 0
        m = c[(c["method"] == method) & (c["train_year"] == t) & (c["eval_year"] == y)
              & (c["sel_range_nm"] == sel_r) & (c["n_ports"] == N) & (c["groups"] == "all")
              & (c["vessel_group"] == grp)]
        return 100 * float(m["feasible_share"].iloc[0]) if len(m) else None
    for method in ("greedy", "volume"):
        for name, (t, y) in {"insample": (TRAIN, TRAIN), "transfer": (TRAIN, EVAL),
                             "oracle": (EVAL, EVAL)}.items():
            v = cell(method, t, y)
            if v is not None:
                f[f"{method}_{name}_pct"] = v
    y = yap[(yap["vessel_group"] == "container") & (yap["eval_year"] == EVAL)
            & (yap["range_nm"] == RANGE)]
    for sid in ("yap4", "yap8"):
        m = y[y["method"] == f"external:{sid}"]
        if len(m):
            for k in ("touch_share_w", "feasible_share_w", "covered_share_w",
                      "touch_share", "feasible_share"):
                if not np.isnan(m[k].iloc[0]):
                    f[f"{sid}_container_{k}_pct"] = 100 * float(m[k].iloc[0])
    if len(own):
        o = own[(own["set_key"] == "external:yap4|-1|0|all") & (own["year"] == EVAL)
                & (own["vessel_group"] == "container")]
        if len(o):
            f["yap4_container_fits_methanol_pct"] = 100 * float(o["fits_methanol"].iloc[0])
            f["yap4_container_fits_hfo_pct"] = 100 * float(o["fits_hfo"].iloc[0])
    e = endur[endur["vessel_group"] == "container"]
    if len(e) and not np.isnan(e["hfo_p50"].iloc[0]):
        f["container_hfo_endurance_p50_nm"] = float(e["hfo_p50"].iloc[0])
        f["container_methanol_endurance_p50_nm"] = float(e["methanol_p50"].iloc[0])
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--train", type=int, default=TRAIN)
    ap.add_argument("--eval", dest="eval_", type=int, default=EVAL)
    ap.add_argument("--range", dest="R", type=int, default=RANGE)
    ap.add_argument("--n", type=int, default=N)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    results = (a.results or out_dir / "results").expanduser().resolve()
    require(out_dir, "coverage.csv", "transfer.csv", "ship_years.csv.gz",
            "redeployment.csv.gz", "vessels.csv", "stops.csv.gz", stage="evaluate.py")
    results.mkdir(parents=True, exist_ok=True)

    cov = pd.read_csv(out_dir / "coverage.csv")
    tr = pd.read_csv(out_dir / "transfer.csv")
    sy = pd.read_csv(out_dir / "ship_years.csv.gz", dtype={"imo": str})
    red = pd.read_csv(out_dir / "redeployment.csv.gz", dtype={"imo": str})
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    stops = pd.read_csv(out_dir / "stops.csv.gz", dtype=str,
                        usecols=["imo", "visit_start", "iso3"])
    regions = pd.read_csv(REGIONS, dtype=str)
    print(f"coverage {len(cov):,}  transfer {len(tr):,}  ship-years {len(sy):,}  "
          f"vessels {len(vessels):,}")

    st = staleness(cov)
    lag = staleness_by_lag(st, a.R)
    terms = transfer_terms(tr)
    crit = criteria(tr, a.train, a.eval_, a.R, a.n)
    v = vessel_endurance(vessels)
    endur = endurance_summary(v)
    rr = required_range(sy)
    cdf = required_range_cdf(sy, a.train, a.eval_, a.R, a.n)
    fe = out_dir / "facts_evaluate.json"
    weighted = fe.exists() and json.loads(fe.read_text()).get("weighted") == "co2"
    if not weighted:
        print("  evaluate ran unweighted: CO2-weighted columns are left out")
    own = own_endurance(sy, v, weighted)
    rl = redeploy_loss(sy, red, a.train, a.R, a.n)
    reg = ship_year_regions(stops, regions)
    del stops
    rg = region_loss(sy, reg, a.train, a.eval_, a.R, a.n)
    yap = yap_comparison(cov)
    bk = backup_table(cov)
    vg = volume_vs_greedy(cov, a.R)
    sens = sensitivity(out_dir, cov, a.train, a.eval_, a.R, a.n)

    tables = {"coverage.csv": cov, "staleness.csv": st, "staleness_by_lag.csv": lag,
              "transfer_terms.csv": terms, "criteria.csv": crit,
              "endurance_summary.csv": endur, "required_range.csv": rr,
              "required_range_cdf.csv": cdf,
              "own_endurance.csv": own, "redeploy_loss.csv": rl, "region_loss.csv": rg,
              "yap_comparison.csv": yap, "backup.csv": bk, "volume_vs_greedy.csv": vg,
              "sensitivity.csv": sens}
    for name, t in tables.items():
        t.to_csv(results / name, index=False)
    for p in sorted(out_dir.glob("coverage_*.csv")):
        shutil.copy(p, results / p.name)
    for p in sorted(out_dir.glob("facts_*.json")):
        shutil.copy(p, results / p.name)

    facts = headline_facts(cov, st, crit, yap, own, endur, a.train, a.eval_, a.R, a.n)
    emit(results, "analysis", facts)
    manifest(results)

    print("\n--- pre-registered test (measurement, then verdict) ---")
    if len(crit):
        print(crit[["method", "vessel_group", "F_t", "F_y", "drop_pp", "panel_drop_pp",
                    "composition_pp", "verdict_group"]].round(3).to_string(index=False))
    print(f"overall: {facts['verdict']}")
    print("\n--- headline facts ---")
    for k, val in facts.items():
        print(f"  {k:<40} {val}")
    print(f"\nwrote {len(tables)} tables to {results}")


if __name__ == "__main__":
    main()
