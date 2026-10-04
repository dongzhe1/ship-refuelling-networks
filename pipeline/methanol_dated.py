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
from facts import emit
from select_ports import load_inputs
from supplement import (SCOPES, endurance_lookup, fuel_intensity, full_rows, green_cols,
                        operating_endurance, scen_tag, scope_weights, set_tables,
                        tank_tonnes)

HERE = Path(__file__).resolve().parent
EVENTS = HERE / "reference" / "methanol_bunkering_events.csv"

YEARS, HEADLINE, PILOT_YEARS, RANGE = "2023-2025", 2025, "2024-2026", 7000
LONG_RANGE = 20000
M1_FRACTION = 0.5
STRICT_DROP_TYPES = ("pilot", "trial")
STRICT_DROP_RECEIVERS = ("small craft",)
STRICT_DROP_CONFIDENCE = ("C",)
GREEN = [("x1", ("methanol", "operating", 1.0, 0.0)),
         ("x2", ("methanol", "operating", 2.0, 0.0))]
BOUNDS, VARIANTS = ("lower", "upper"), ("all", "strict")


def read_events(path: Path = EVENTS) -> pd.DataFrame:
    e = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    return e[e["fuel"] != "ethanol"].reset_index(drop=True)


def date_window(date: str, precision: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if precision == "day":
        t = pd.Timestamp(date)
        return t, t
    if precision == "month":
        t = pd.Timestamp(date + "-01")
        return t, t + pd.offsets.MonthEnd(0)
    if precision == "year":
        return pd.Timestamp(date + "-01-01"), pd.Timestamp(date + "-12-31")
    if precision == "before":
        t = pd.Timestamp(date + "-01") if len(date) == 7 else pd.Timestamp(date + "-01-01")
        return t, t
    raise ValueError(f"unknown date precision {precision!r}")


def strict_mask(e: pd.DataFrame) -> pd.Series:
    return ~(e["event_type"].isin(STRICT_DROP_TYPES)
             | e["receiving_type"].isin(STRICT_DROP_RECEIVERS)
             | e["confidence"].isin(STRICT_DROP_CONFIDENCE))


def node_dates(events: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    node_of = nodes.set_index("anchorage_id")["node_id"]
    w = events.apply(lambda r: date_window(r["date"], r["date_precision"]), axis=1)
    e = events.assign(dmin=[a for a, _ in w], dmax=[b for _, b in w], strict=strict_mask(events))
    rows, missing = [], set()
    for r in e.itertuples():
        ids = [a for a in r.anchorage_ids.split(";") if a]
        found = {node_of[a] for a in ids if a in node_of.index}
        missing |= {a for a in ids if a not in node_of.index}
        for n in found:
            rows.append({"node_id": n, "port": r.port, "event_id": r.event_id,
                         "dmin": r.dmin, "dmax": r.dmax, "strict": r.strict})
    if missing:
        print(f"  anchorages not in the node table (ignored): {', '.join(sorted(missing))}")
    d = pd.DataFrame(rows, columns=["node_id", "port", "event_id", "dmin", "dmax", "strict"])
    out = []
    for variant in VARIANTS:
        x = d if variant == "all" else d[d["strict"]]
        g = x.groupby("node_id").agg(port=("port", "first"), available_from=("dmin", "min"),
                                     certain_by=("dmax", "min"))
        out.append(g.reset_index().assign(variant=variant))
    return pd.concat(out, ignore_index=True)


def dated_sets(nd: pd.DataFrame, years) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        v = nd[nd["variant"] == variant]
        for y in years:
            start, end = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            for bound in BOUNDS:
                m = v["certain_by"] < start if bound == "lower" else v["available_from"] <= end
                s = v[m].sort_values("node_id")
                rows.append({"variant": variant, "bound": bound, "year": int(y),
                             "n_nodes": len(s), "nodes": ";".join(s["node_id"]),
                             "ports": ";".join(s["port"])})
    return pd.DataFrame(rows)


def fleet_rows(fy: pd.DataFrame, R: int, meta: dict) -> list[dict]:
    rows = []
    groups = [("all", fy)] + [(g, p) for g, p in fy.groupby("vessel_group")]
    for grp, p in groups:
        W = p["w_total"].sum()
        if W <= 0:
            continue
        row = {**meta, "vessel_group": grp, "n_full": len(p),
               "touch_w": float(p.loc[p["touches"].astype(bool), "w_total"].sum() / W),
               "feasible_w": float(p.loc[p["req_lb"] <= R, "w_total"].sum() / W),
               "covered_lb_w": float(p[f"lb_w_{R}"].sum() / W),
               "covered_ub_w": float(p[f"ub_w_{R}"].sum() / W)}
        for name, (fuel, basis, mult, res) in GREEN:
            tag = scen_tag(fuel, basis, mult, res)
            k = p[p[f"{basis}_{fuel}_nm"].notna()]
            Wk = k["w_total"].sum()
            g = green_cols(k, tag)
            row[f"green_{name}_lb_w"], row[f"green_{name}_ub_w"] = g["green_lb_w"], g["green_ub_w"]
            for m in ["p", "f"] + [f"{sc}{k}" for sc in SCOPES for k in ("", "_f", "_o")]:
                for b in ("lb", "ub"):
                    row[f"green_{name}_{m}_{b}_w"] = g[f"green_{m}_{b}_w"]
            if name == "x1":
                row["touch_known_w"] = float(k.loc[k["touches"].astype(bool), "w_total"].sum()
                                             / Wk) if Wk > 0 else np.nan
                row["feasible_known_w"] = float(k.loc[k["req_lb"] <= R, "w_total"].sum()
                                                / Wk) if Wk > 0 else np.nan
                row["n_known_tank"] = len(k)
        rows.append(row)
    return rows


def pilot_rows(sy: pd.DataFrame, R: int, meta: dict) -> dict:
    full = sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]
    req = full["req_lb"].to_numpy(float)
    return {**meta, "n_ship_years": int(len(sy)), "n_full": int(len(full)),
            "n_ships": int(full["imo"].nunique()),
            "touch_share": float(full["touches"].astype(bool).mean()) if len(full) else np.nan,
            "feasible_share": float((req <= R).mean()) if len(req) else np.nan,
            "feasible_long_share": float((req <= LONG_RANGE).mean()) if len(req) else np.nan,
            "median_req_lb": float(np.median(req)) if len(req) else np.nan}


def readings(fleet: pd.DataFrame, pilot: pd.DataFrame, headline: int, years) -> dict:
    f = {}
    c = fleet[(fleet["vessel_group"] == "container") & (fleet["year"] == headline)]

    def val(variant, bound, col):
        r = c[(c["variant"] == variant) & (c["bound"] == bound)]
        return float(r[col].iloc[0]) if len(r) else np.nan
    for variant in VARIANTS:
        cov_ub, touch_lb = val(variant, "upper", "covered_ub_w"), val(variant, "lower", "touch_w")
        f[f"m1_{variant}_covered_ub_pct"] = 100 * cov_ub
        f[f"m1_{variant}_touch_lb_pct"] = 100 * touch_lb
        f[f"m1_{variant}_reading"] = (
            "not estimable" if not (np.isfinite(cov_ub) and np.isfinite(touch_lb)) else
            "robust on the real network" if cov_ub < M1_FRACTION * touch_lb else
            "state the contrast as a range")
        for bound in BOUNDS:
            for col in ("touch_w", "feasible_w", "covered_lb_w", "green_x1_ub_w",
                        "green_x2_ub_w", "touch_known_w", "green_x1_p_ub_w", "green_x2_p_ub_w",
                        "green_x1_eu_ub_w", "green_x2_eu_ub_w", "green_x1_f_ub_w",
                        "green_x2_f_ub_w", "green_x1_eu_o_ub_w", "green_x2_eu_o_ub_w"):
                f[f"{variant}_{bound}_{col[:-2]}_pct"] = 100 * val(variant, bound, col)
    g2, tk = val("all", "upper", "green_x2_ub_w"), val("all", "lower", "touch_known_w")
    f["m2_calling_over_green_x2"] = tk / g2 if g2 > 0 else np.nan
    g2p = val("all", "upper", "green_x2_p_ub_w")
    f["m2p_calling_over_green_x2"] = tk / g2p if g2p > 0 else np.nan
    first = min(years)
    c0 = fleet[(fleet["vessel_group"] == "container") & (fleet["year"] == first)
               & (fleet["variant"] == "all") & (fleet["bound"] == "upper")]
    c1 = c[(c["variant"] == "all") & (c["bound"] == "upper")]
    if len(c0) and len(c1):
        f["m3_touch_change_pp"] = 100 * float(c1["touch_w"].iloc[0] - c0["touch_w"].iloc[0])
        f["m3_covered_change_pp"] = 100 * float(c1["covered_lb_w"].iloc[0]
                                                - c0["covered_lb_w"].iloc[0])
    if len(pilot):
        for (pdir, variant, bound), r in pilot[pilot["period"] == "pool"].groupby(
                ["pilot", "variant", "bound"]):
            tag = f"m4_{Path(pdir).name}_{variant}_{bound}"
            f[f"{tag}_feasible_pct"] = 100 * float(r["feasible_share"].iloc[0])
            f[f"{tag}_touch_pct"] = 100 * float(r["touch_share"].iloc[0])
            f[f"{tag}_median_req_nm"] = float(r["median_req_lb"].iloc[0])
    return {k: v for k, v in f.items() if isinstance(v, str) or np.isfinite(v)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--events", type=Path, default=EVENTS)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--years", default=YEARS)
    ap.add_argument("--headline", type=int, default=HEADLINE)
    ap.add_argument("--range", dest="R", type=int, default=RANGE)
    ap.add_argument("--pilot", default="", help="pilot directories, comma-separated")
    ap.add_argument("--pilot-years", default=PILOT_YEARS)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "stops.csv.gz", "nodes.csv", "vessels.csv", stage="vessels.py / evaluate.py")
    results = out_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    years, pyears = parse_years(a.years), parse_years(a.pilot_years)
    R = a.R

    nodes = pd.read_csv(out_dir / "nodes.csv", dtype={"anchorage_id": str, "node_id": str})
    nd = node_dates(read_events(a.events), nodes)
    sets = dated_sets(nd, sorted(set(years) | set(pyears)))
    nd.to_csv(results / "methanol_dated_nodes.csv", index=False)
    sets.to_csv(results / "methanol_dated_sets.csv", index=False)
    print("dated networks (nodes):")
    print(sets.pivot_table(index=["variant", "bound"], columns="year", values="n_nodes")
          .to_string())

    stops, nodes, labels = load_inputs(out_dir, extra=("sea_hours",))
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    group_of = vessels.set_index("imo")["group"]
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, a.calibration)
    stops["leg_fuel_t"] = stops["leg_co2_t"] / emissions.CO2_PER_FUEL_T
    endur_vy, endur_v = endurance_lookup(operating_endurance(vessels, fuel_intensity(stops)),
                                         vessels)
    prep = chains.prepare(stops, labels, weight_col="leg_co2_t")
    scopes = scope_weights(prep, stops, nodes)
    del stops
    cache: dict[str, pd.DataFrame] = {}
    fleet = []
    for r in sets[sets["year"].isin(years)].itertuples():
        if r.nodes not in cache:
            cache[r.nodes] = set_tables(prep, chains.refuel_mask(prep, r.nodes.split(";")
                                                                 if r.nodes else []),
                                        endur_vy, endur_v, ranges=(R,),
                                        scenarios=[sc for _, sc in GREEN], scopes=scopes,
                                        tanks=tank_tonnes(vessels))
        fy = full_rows(cache[r.nodes], r.year, group_of)
        fleet += fleet_rows(fy, R, {"variant": r.variant, "bound": r.bound, "year": r.year,
                                    "n_nodes": r.n_nodes})
    fleet = pd.DataFrame(fleet)
    del cache, prep
    fleet.to_csv(results / "methanol_dated.csv", index=False)
    show = fleet[fleet["vessel_group"] == "container"]
    print("\ncontainer ships, CO2-weighted (%):")
    print((show.set_index(["variant", "bound", "year"])[
        ["n_nodes", "touch_w", "feasible_w", "covered_lb_w", "covered_ub_w", "green_x1_ub_w",
         "green_x2_ub_w"]] * [1, 100, 100, 100, 100, 100, 100]).round(1).to_string())

    pilot = []
    for pdir in [Path(p).expanduser().resolve() for p in a.pilot.split(",") if p]:
        if not (pdir / "stops.csv.gz").exists():
            print(f"  pilot {pdir}: no stops.csv.gz, skipped")
            continue
        pstops, _, plabels = load_inputs(pdir)
        pprep = chains.prepare(pstops, plabels)
        pcache: dict[str, pd.DataFrame] = {}
        for variant in VARIANTS:
            for bound in BOUNDS:
                pooled = []
                for y in pyears:
                    s = sets[(sets["variant"] == variant) & (sets["bound"] == bound)
                             & (sets["year"] == y)].iloc[0]
                    if s["nodes"] not in pcache:
                        refuel = chains.refuel_mask(pprep, s["nodes"].split(";") if s["nodes"] else [])
                        seg = chains.segments(pprep, refuel)
                        pcache[s["nodes"]] = chains.ship_years(pprep, seg, refuel, (R, LONG_RANGE))
                    sy = pcache[s["nodes"]]
                    sy = sy[sy["year"] == y]
                    pooled.append(sy)
                    pilot.append(pilot_rows(sy, R, {"pilot": pdir.name, "variant": variant,
                                                    "bound": bound, "period": str(y),
                                                    "n_nodes": int(s["n_nodes"])}))
                pilot.append(pilot_rows(pd.concat(pooled, ignore_index=True), R,
                                        {"pilot": pdir.name, "variant": variant, "bound": bound,
                                         "period": "pool", "n_nodes": -1}))
    pilot = pd.DataFrame(pilot)
    pilot.to_csv(results / "methanol_dated_pilot.csv", index=False)
    if len(pilot):
        print("\npilot ships, pooled:")
        print(pilot[pilot["period"] == "pool"][["pilot", "variant", "bound", "n_full",
                                               "touch_share", "feasible_share",
                                               "feasible_long_share", "median_req_lb"]]
              .round(3).to_string(index=False))

    facts = readings(fleet, pilot, a.headline, years)
    emit(results, "methanol_dated", {"headline_year": a.headline, "range_nm": R, **facts})
    print("\nreadings:")
    for k, v in facts.items():
        if k.startswith("m"):
            print(f"  {k:<48} {v}")


if __name__ == "__main__":
    main()
