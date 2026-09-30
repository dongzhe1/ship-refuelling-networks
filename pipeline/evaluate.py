from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
import emissions
from common import (fork_pool, DEFAULT_JOBS, parse_floats, parse_ints, parse_years, require,
                    tagged)
from facts import emit
from select_ports import load_inputs

RANGES = "2000,3000,5000,7000,10000,15000,20000"
BASELINE_KEY = "all|-1|0|all"

META_COLS = ["set_key", "method", "train_year", "sel_range_nm", "groups", "n_ports"]
COV_COLS = META_COLS + ["eval_year", "vessel_group", "range_nm", "n_active", "n_full",
                        "feasible_share", "touch_share", "covered_share", "median_req_lb",
                        "feasible_share_w", "touch_share_w", "covered_share_w", "w_full"]
TRANSFER_COLS = META_COLS + ["eval_year", "vessel_group", "range_nm", "n_t", "F_t",
                             "n_y", "F_y", "n_panel", "F_panel_t", "F_panel_y",
                             "n_entry", "F_entry", "n_exit", "F_exit"]
SY_COLS = ["imo", "year", "vessel_group", "n_stops", "dist_nm", "w_total", "n_breaks",
           "touches", "req_lb", "req_complete", "n_censored"]


def is_backup(method) -> bool:
    return str(method).startswith("backup")


def build_sets(port_sets: pd.DataFrame, ns, labels, only: str = "",
               baseline: bool = True) -> list[dict]:
    sets = []
    pat = re.compile(only) if only else None
    for key, g in port_sets.groupby("set_key", sort=True):
        if pat and not pat.search(key):
            continue
        meta = g.iloc[0]
        g = g.sort_values("rank")
        top = int(g["rank"].max())
        if str(meta["method"]).startswith("external"):
            cuts = [top]
        elif is_backup(meta["method"]):
            n0 = int(g["initial"].astype(str).eq("True").sum()) if "initial" in g else 0
            cuts = list(range(max(n0, 1), top + 1))
        else:
            cuts = [n for n in ns if n <= top]
        for n in cuts:
            ids = g.loc[g["rank"] <= n, "node_id"].astype(str).tolist()
            if not ids:
                continue
            sets.append({"set_key": key, "method": meta["method"],
                         "train_year": int(meta["train_year"]),
                         "sel_range_nm": int(meta["sel_range_nm"]),
                         "groups": meta["groups"], "n_ports": len(set(ids)),
                         "nodes": ids})
    if baseline:
        sets.append({"set_key": BASELINE_KEY, "method": "all", "train_year": -1,
                     "sel_range_nm": 0, "groups": "all", "n_ports": len(labels),
                     "nodes": list(labels)})
    return sets


def _share(num, den):
    return float(num / den) if den > 0 else np.nan


def coverage_rows(sy: pd.DataFrame, meta: dict, ranges, years, weighted=False) -> list[dict]:
    rows = []
    base = sy[sy["year"].isin(years) & sy["req_lb"].notna()]
    for grp_name, part in [("all", base)] + list(base.groupby("vessel_group")):
        for y, py in part.groupby("year"):
            full = py[py["n_breaks"] == 0]
            dist = full["dist_nm"].sum()
            w = full["w_total"].to_numpy(float) if weighted else None
            wsum = float(w.sum()) if weighted else np.nan
            for R in ranges:
                feas = (full["req_lb"] <= R).to_numpy()
                row = {**{k: meta[k] for k in META_COLS},
                       "eval_year": int(y), "vessel_group": grp_name, "range_nm": int(R),
                       "n_active": int(len(py)), "n_full": int(len(full)),
                       "feasible_share": float(feas.mean()) if len(full) else np.nan,
                       "touch_share": float(full["touches"].mean()) if len(full) else np.nan,
                       "covered_share": _share(full[f"covered_{int(R)}"].sum(), dist),
                       "median_req_lb": float(full["req_lb"].median()) if len(full) else np.nan,
                       "feasible_share_w": np.nan, "touch_share_w": np.nan,
                       "covered_share_w": np.nan, "w_full": wsum}
                if weighted:
                    row["feasible_share_w"] = _share(w[feas].sum(), wsum)
                    row["touch_share_w"] = _share(w[full["touches"].to_numpy(bool)].sum(), wsum)
                    row["covered_share_w"] = _share(full[f"covered_w_{int(R)}"].sum(), wsum)
                rows.append(row)
    return rows


def transfer_rows(sy: pd.DataFrame, meta: dict, ranges, years) -> list[dict]:
    t = meta["train_year"]
    if t < 0:
        return []
    full = sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]
    rows = []

    def mean(s):
        return float(s.mean()) if len(s) else np.nan

    for grp_name, part in [("all", full)] + list(full.groupby("vessel_group")):
        at = part[part["year"] == t].set_index("imo")["req_lb"]
        if at.empty:
            continue
        for y in years:
            if y == t:
                continue
            ay = part[part["year"] == y].set_index("imo")["req_lb"]
            if ay.empty:
                continue
            panel = at.index.intersection(ay.index)
            entry = ay.index.difference(at.index)
            exit_ = at.index.difference(ay.index)
            for R in ranges:
                ft, fy = at <= R, ay <= R
                rows.append({
                    **{k: meta[k] for k in META_COLS},
                    "eval_year": int(y), "vessel_group": grp_name, "range_nm": int(R),
                    "n_t": int(len(at)), "F_t": mean(ft), "n_y": int(len(ay)), "F_y": mean(fy),
                    "n_panel": int(len(panel)), "F_panel_t": mean(ft.loc[panel]),
                    "F_panel_y": mean(fy.loc[panel]), "n_entry": int(len(entry)),
                    "F_entry": mean(fy.loc[entry]), "n_exit": int(len(exit_)),
                    "F_exit": mean(ft.loc[exit_]),
                })
    return rows


def redeployment(prep: chains.Prepared, train_years, years) -> pd.DataFrame:
    vn = (pd.DataFrame({"imo": prep.imo, "year": prep.year, "node": prep.node})
          .query("node >= 0").drop_duplicates())
    size = vn.groupby(["imo", "year"]).size()
    out = []
    for t in train_years:
        a = vn[vn["year"] == t][["imo", "node"]]
        for y in years:
            if y == t:
                continue
            b = vn[vn["year"] == y][["imo", "node"]]
            shared = a.merge(b, on=["imo", "node"]).groupby("imo").size()
            both = a["imo"].drop_duplicates()
            both = both[both.isin(set(b["imo"]))]
            if both.empty:
                continue
            na = size.xs(t, level="year").reindex(both).to_numpy()
            nb = size.xs(y, level="year").reindex(both).to_numpy()
            ns = shared.reindex(both).fillna(0).to_numpy()
            out.append(pd.DataFrame({"imo": both.to_numpy(), "train_year": t,
                                     "year": y, "n_nodes_t": na, "n_nodes_y": nb,
                                     "n_shared": ns.astype(int),
                                     "jaccard": ns / (na + nb - ns)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["imo", "train_year", "year", "n_nodes_t", "n_nodes_y", "n_shared",
                 "jaccard"])


_PREP = None
_GROUP_OF = None
_ARGS = None


def keeps_ship_years(meta, keep_ns) -> bool:
    if keep_ns is None:
        return False
    m = str(meta["method"])
    return (keep_ns == "all" or m == "all" or m.startswith("external")
            or is_backup(m) or meta["n_ports"] in keep_ns)


def _eval_one(meta):
    ranges, years, keep_ns = _ARGS
    refuel = chains.refuel_mask(_PREP, meta["nodes"])
    seg = chains.segments(_PREP, refuel)
    sy = chains.ship_years(_PREP, seg, refuel, ranges)
    sy["vessel_group"] = sy["imo"].map(_GROUP_OF).fillna("unknown")
    cov = coverage_rows(sy, meta, ranges, years, weighted=_PREP.weighted)
    tr = transfer_rows(sy, meta, ranges, years)
    s = None
    if keeps_ship_years(meta, keep_ns):
        s = sy.loc[sy["year"].isin(years), SY_COLS].copy()
        s.insert(0, "n_ports", meta["n_ports"])
        s.insert(0, "set_key", meta["set_key"])
    return cov, tr, s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--sets", default="port_sets.csv")
    ap.add_argument("--stops", default="stops.csv.gz")
    ap.add_argument("--nodes", default="nodes.csv")
    ap.add_argument("--ns", default="5,10,20,50")
    ap.add_argument("--ranges", default=RANGES)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--weight", choices=["none", "co2"], default="none")
    ap.add_argument("--calibration", type=Path,
                    default=Path(os.environ["GN_CALIBRATION"])
                    if os.environ.get("GN_CALIBRATION") else None,
                    help="the type_calibration.csv (read only)")
    ap.add_argument("--slow-as-break", action="store_true")
    ap.add_argument("--only", default="", help="regex on set_key: evaluate a subset")
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--no-ship-years", action="store_true")
    ap.add_argument("--ship-years-ns", default="20",
                    help="write vessel-years only for rankings cut at these N "
                         "(external, backup and baseline always); 'all' for every set")
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    ranges, years, ns = parse_floats(a.ranges), parse_years(a.years), parse_ints(a.ns)

    require(out_dir, a.sets, stage="select_ports.py")
    extra = ("sea_hours",) if a.weight == "co2" else ()
    stops, nodes, labels = load_inputs(out_dir, a.stops, a.nodes,
                                       slow_as_break=a.slow_as_break, extra=extra)
    vpath = out_dir / "vessels.csv"
    vessels = pd.read_csv(vpath, dtype={"imo": str}) if vpath.exists() else None
    group_of = vessels.set_index("imo")["group"] if vessels is not None else pd.Series(dtype=str)
    if vessels is None:
        print("  no vessels.csv: every vessel is group 'unknown'")
    weight_col = None
    if a.weight == "co2":
        if vessels is None:
            sys.exit("--weight co2 needs vessels.csv (run vessels.py)")
        stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, a.calibration)
        weight_col = "leg_co2_t"
        ok = stops["leg_ok"].to_numpy()
        print(f"  CO2 on valid legs: {stops.loc[ok, 'leg_co2_t'].sum() / 1e6:,.1f} Mt; "
              f"{100 * (stops.loc[ok, 'leg_co2_t'] > 0).mean():.1f}% of valid legs weighted")
    prep = chains.prepare(stops, labels, weight_col=weight_col)
    del stops
    port_sets = pd.read_csv(out_dir / a.sets, dtype={"node_id": str})
    sets = build_sets(port_sets, ns, labels, a.only)
    print(f"{prep.n:,} stops, {len(np.unique(prep.vessel)):,} vessels; "
          f"{len(sets)} set(s) x {len(years)} year(s) x {len(ranges)} range(s)"
          f"{'; CO2-weighted' if prep.weighted else ''}")

    global _PREP, _GROUP_OF, _ARGS
    keep_ns = None if a.no_ship_years else (
        "all" if a.ship_years_ns.strip() == "all" else set(parse_ints(a.ship_years_ns)))
    _PREP, _GROUP_OF, _ARGS = prep, group_of, (ranges, years, keep_ns)
    n = max(1, min(a.jobs, len(sets)))
    pool = fork_pool(n)
    if pool is not None:
        with pool:
            results = pool.map(_eval_one, sets, chunksize=1)
    else:
        results = [_eval_one(s) for s in sets]

    name = lambda f: tagged(f, a.tag)
    cov = pd.DataFrame([r for c, _, _ in results for r in c], columns=COV_COLS)
    tr = pd.DataFrame([r for _, t, _ in results for r in t], columns=TRANSFER_COLS)
    cov.to_csv(out_dir / name("coverage.csv"), index=False)
    tr.to_csv(out_dir / name("transfer.csv"), index=False)
    kept = [s for _, _, s in results if s is not None]
    if kept:
        pd.concat(kept, ignore_index=True).to_csv(
            out_dir / name("ship_years.csv.gz"), index=False, compression="gzip")
    train_years = sorted({s["train_year"] for s in sets if s["train_year"] >= 0})
    red = redeployment(prep, train_years, years)
    red.to_csv(out_dir / name("redeployment.csv.gz"), index=False, compression="gzip")

    g = cov[(cov["method"] == "greedy") & (cov["vessel_group"] == "all")
            & (cov["groups"] == "all")]
    if not g.empty:
        R = g["sel_range_nm"].min()
        nn = 20 if (g["n_ports"] == 20).any() else g["n_ports"].max()
        m = g[(g["sel_range_nm"] == R) & (g["n_ports"] == nn) & (g["range_nm"] == R)]
        if not m.empty:
            print(f"\nfeasible share, greedy N={nn}, R={R} nm (rows: chosen on; cols: judged on)")
            print(m.pivot_table(index="train_year", columns="eval_year",
                                values="feasible_share").round(3).to_string())
    base = cov[(cov["set_key"] == BASELINE_KEY) & (cov["vessel_group"] == "all")]
    full_share = (base["n_full"] / base["n_active"]).mean() if len(base) else 0.0
    emit(out_dir, name("evaluate"), {
        "sets": int(len(sets)),
        "ship_years_active": int(base.groupby("eval_year")["n_active"].first().sum())
        if len(base) else 0,
        "fully_observed_pct": 100.0 * float(full_share),
        "baseline_median_req_nm": float(base["median_req_lb"].median()) if len(base) else 0.0,
        "weighted": a.weight, "slow_as_break": bool(a.slow_as_break),
        "nodes": a.nodes, "sets_file": a.sets,
    })
    print(f"\nwrote {name('coverage.csv')} ({len(cov):,} rows), {name('transfer.csv')} "
          f"({len(tr):,}), {name('redeployment.csv.gz')} ({len(red):,})")


if __name__ == "__main__":
    main()
