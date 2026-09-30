from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
import emissions
from common import parse_years, require
from evaluate import build_sets
from facts import emit
from loss_by_gap import GROUPS, TRANSITION_COLS, strata
from pilot import wilson
from select_ports import load_inputs
from supplement import (endurance_lookup, fuel_intensity, operating_endurance, scen_tag,
                        set_tables)

CUT = 0.5
SCEN = ("methanol", "operating", 1.0, 0.0)
VERSIONS = {"r_registered": ("share_r", False), "r_common": ("share_r", True),
            "own_common": ("share_own", True)}


def ship_year_shares(t: pd.DataFrame, R: int, group_of) -> pd.DataFrame:
    tag = scen_tag(*SCEN)
    f = t[(t["n_breaks"] == 0) & t["req_lb"].notna() & (t["w_total"] > 0)].copy()
    return pd.DataFrame({
        "imo": f["imo"].astype(str), "year": f["year"].astype(int),
        "vessel_group": f["imo"].map(group_of).fillna("unknown"),
        "known": f["operating_methanol_nm"].notna(),
        "share_r": f[f"plb_R{int(R)}"] / f["w_total"],
        "share_own": f[f"plb_{tag}"] / f["w_total"]})


def transitions(by_t: dict[int, pd.DataFrame], stratum: pd.Series, years) -> pd.DataFrame:
    rows = []
    for version, (col, known_only) in VERSIONS.items():
        for t, d in sorted(by_t.items()):
            d = d[d["known"]] if known_only else d
            d = d.assign(stratum=d["imo"].map(stratum).fillna("unknown"),
                         low=d[col] < CUT)
            base = d[(d["year"] == t) & (d[col] >= CUT)].set_index("imo")
            for y in years:
                if y <= t:
                    continue
                ev = d[d["year"] == y]
                f_low = ev.groupby("stratum")["low"].mean()
                ev = ev.set_index("imo")
                pop = base[["vessel_group", "stratum"]].join(ev[["low"]], how="inner")
                pop["indep"] = pop["stratum"].map(f_low)
                for g in GROUPS:
                    p = pop if g == "all" else pop[pop["vessel_group"] == g]
                    n = len(p)
                    if n == 0:
                        continue
                    share = float(p["low"].mean())
                    lo, hi = wilson(share, n)
                    rows.append({"version": version, "train_year": int(t), "eval_year": int(y),
                                 "gap": int(y - t), "vessel_group": g, "n": n,
                                 "lost": int(p["low"].sum()), "lost_share": share,
                                 "lo": lo, "hi": hi, "indep_share": float(p["indep"].mean())})
    return pd.DataFrame(rows, columns=["version", *TRANSITION_COLS])


def summary(tr: pd.DataFrame, train: int, eval_: int) -> dict:
    f = {}
    for version in VERSIONS:
        for g in ("all", "container"):
            a = tr[(tr["version"] == version) & (tr["vessel_group"] == g)]
            if not len(a):
                continue
            pre = f"{version}_{g}"
            by_gap = a.groupby("gap").apply(lambda x: x["lost"].sum() / x["n"].sum(),
                                            include_groups=False)
            for k, v in by_gap.items():
                f[f"{pre}_gap{k}_lost_pct"] = 100 * float(v)
            r = a[(a["train_year"] == train) & (a["eval_year"] == eval_)]
            if len(r):
                r = r.iloc[0]
                f[f"{pre}_headline_lost_pct"] = 100 * r["lost_share"]
                f[f"{pre}_headline_n"] = int(r["n"])
                f[f"{pre}_headline_indep_pct"] = 100 * r["indep_share"]
                if 1 in by_gap.index:
                    f[f"{pre}_headline_excess_over_gap1_pp"] = 100 * (r["lost_share"] - by_gap.loc[1])
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--range", dest="R", type=int, default=7000)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--train", type=int, default=2019)
    ap.add_argument("--eval", dest="eval_", type=int, default=2024)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "stops.csv.gz", "nodes.csv", "vessels.csv", "port_sets.csv", stage="select_ports.py")
    years = parse_years(a.years)

    stops, nodes, labels = load_inputs(out_dir, extra=("sea_hours",))
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    group_of = vessels.set_index("imo")["group"]
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, a.calibration)
    stops["leg_fuel_t"] = stops["leg_co2_t"] / emissions.CO2_PER_FUEL_T
    endur_vy, endur_v = endurance_lookup(operating_endurance(vessels, fuel_intensity(stops)),
                                         vessels)
    prep = chains.prepare(stops, labels, weight_col="leg_co2_t")
    del stops
    port_sets = pd.read_csv(out_dir / "port_sets.csv", dtype={"node_id": str})
    built = build_sets(port_sets, [a.n], labels)
    by_t = {}
    for t in years:
        key = f"greedy|{t}|{a.R}|all"
        s = next((x for x in built if x["set_key"] == key and x["n_ports"] == a.n), None)
        if s is None:
            print(f"  {key} (N={a.n}) not in port_sets.csv, skipped")
            continue
        tab = set_tables(prep, chains.refuel_mask(prep, s["nodes"]), endur_vy, endur_v,
                         ranges=(), scenarios=[SCEN], strand_R=a.R)
        by_t[t] = ship_year_shares(tab, a.R, group_of)
        print(f"  {key}: {len(by_t[t]):,} fully observed ship-years")
    tr = transitions(by_t, strata(vessels), years)
    res = out_dir / "results"
    res.mkdir(exist_ok=True)
    tr.to_csv(res / "loss_by_gap_dualfuel.csv", index=False)
    f = summary(tr, a.train, a.eval_)
    emit(res, "loss_by_gap_dualfuel", {k: v for k, v in f.items() if np.isfinite(v)})
    pd.set_option("display.width", 160)
    for version in VERSIONS:
        x = tr[(tr["version"] == version) & (tr["vessel_group"] == "all")]
        if x.empty:
            print(f"\n{version}: no ship at or above one half in one year and fully "
                  f"observed in a later one")
            continue
        print(f"\n{version}: lost share by starting year (rows) and gap (columns)")
        print(x.pivot_table(index="train_year", columns="gap", values="lost_share").round(3).to_string())
    for k, v in f.items():
        if "headline" in k:
            print(f"  {k:<48} {v:.2f}" if isinstance(v, float) else f"  {k:<48} {v}")


if __name__ == "__main__":
    main()
