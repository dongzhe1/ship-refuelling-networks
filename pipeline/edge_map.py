from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
import emissions
from common import require
from evaluate import build_sets
from facts import emit
from select_ports import load_inputs
from supplement import (endurance_lookup, fuel_intensity, operating_endurance, scen_tag,
                        set_tables)

SCEN = ("methanol", "operating", 1.0, 0.0)
SETS = [("yap4", "external:yap4|-1|0|all", None),
        ("greedy_large", "greedy|{y}|7000|all", "large"),
        ("greedy_small", "greedy|{y}|7000|all", "small"),
        ("dualfuel_small", "dualfuel|{y}|7000|all", "small"),
        ("dualfuel_large", "dualfuel|{y}|7000|all", "large"),
        ("volume_large", "volume|{y}|0|all", "large")]
PORT_SET_FILES = ("port_sets.csv", "port_sets_dualfuel.csv")


def leg_table(prep, refuel, t, year, group, group_of) -> pd.DataFrame:
    tag = scen_tag(*SCEN)
    fuel, basis, mult, res = SCEN
    keep = t[(t["year"] == year) & (t["n_breaks"] == 0) & t["req_lb"].notna()
             & t[f"{basis}_{fuel}_nm"].notna()].copy()
    keep = keep[keep["imo"].map(group_of).fillna("unknown") == group]
    sy = pd.DataFrame({"vessel": prep.vessel, "year": prep.year})
    sy = sy.merge(keep[["vessel", "year", f"{basis}_{fuel}_nm"]], on=["vessel", "year"],
                  how="left")
    E = sy[f"{basis}_{fuel}_nm"].to_numpy(float) * mult * (1 - res)
    d0, anchored = chains.leg_start(prep, refuel)
    f = chains.fraction_within(prep, d0, anchored, E, anchored_only=False)
    inside = ~np.isnan(E)
    prev = np.r_[-1, prep.node[:-1]]
    legs = pd.DataFrame({"a": prev, "b": prep.node, "w": prep.leg_w, "g": f * prep.leg_w})
    legs = legs[inside]
    check = {"w_total": float(keep["w_total"].sum()),
             "w_green": float(keep[f"pub_{tag}"].sum()),
             "w_legs": float(legs["w"].sum()), "w_green_legs": float(legs["g"].sum())}
    return legs, check


def green_share_value(res: Path, key: str, n, year: int, group: str) -> float:
    p = res / "green_share.csv"
    if not p.exists():
        return float("nan")
    g = pd.read_csv(p)
    g = g[(g["set_key"] == key) & (g["year"] == year) & (g["vessel_group"] == group)
          & (g["fuel"] == SCEN[0]) & (g["basis"] == SCEN[1]) & (g["tank_mult"] == SCEN[2])
          & (g["reserve"] == SCEN[3])]
    if n is not None:
        g = g[g["label"].astype(str).str.contains(rf"\b{n}\b")]
    return float(g["green_p_ub_w"].iloc[0]) if len(g) == 1 else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--group", default="container")
    ap.add_argument("--n-large", type=int, default=20)
    ap.add_argument("--n-small", type=int, default=10)
    ap.add_argument("--cover", type=float, default=0.95)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "stops.csv.gz", "nodes.csv", "vessels.csv", "port_sets.csv", stage="select_ports.py")

    stops, nodes, labels = load_inputs(out_dir, extra=("sea_hours",))
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    group_of = vessels.set_index("imo")["group"]
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, a.calibration)
    stops["leg_fuel_t"] = stops["leg_co2_t"] / emissions.CO2_PER_FUEL_T
    endur_vy, endur_v = endurance_lookup(operating_endurance(vessels, fuel_intensity(stops)),
                                         vessels)
    prep = chains.prepare(stops, labels, weight_col="leg_co2_t")
    del stops
    ps = pd.concat([pd.read_csv(out_dir / f, dtype={"node_id": str})
                    for f in PORT_SET_FILES if (out_dir / f).exists()], ignore_index=True)
    ns = {"large": a.n_large, "small": a.n_small}
    built = build_sets(ps, sorted(set(ns.values())), labels)
    res = out_dir / "results"
    res.mkdir(exist_ok=True)

    frames, facts, members = [], {}, {}
    for name, key, size in SETS:
        key = key.format(y=a.year)
        n = ns.get(size)
        s = next((x for x in built if x["set_key"] == key and (n is None or x["n_ports"] == n)),
                 None)
        if s is None:
            print(f"  {name}: {key} (N={n}) not in {', '.join(PORT_SET_FILES)}, skipped")
            continue
        refuel = chains.refuel_mask(prep, s["nodes"])
        t = set_tables(prep, refuel, endur_vy, endur_v, ranges=(), scenarios=[SCEN])
        legs, chk = leg_table(prep, refuel, t, a.year, a.group, group_of)
        share = chk["w_green"] / chk["w_total"] if chk["w_total"] > 0 else float("nan")
        ref = green_share_value(res, key, n, a.year, a.group)
        drawable = (legs["a"] >= 0) & (legs["b"] >= 0) & (legs["a"] != legs["b"])
        lo, hi = np.minimum(legs["a"], legs["b"]), np.maximum(legs["a"], legs["b"])
        agg = (legs[drawable].assign(a=lo[drawable], b=hi[drawable])
               .groupby(["a", "b"]).agg(w=("w", "sum"), w_green=("g", "sum"),
                                        n_legs=("w", "size")).reset_index())
        agg["set"] = name
        frames.append(agg)
        members[name] = set(str(x) for x in s["nodes"])
        facts.update({
            f"{name}_n_nodes": len(s["nodes"]),
            f"{name}_share_pct": 100 * share,
            f"{name}_green_share_csv_pct": 100 * ref,
            f"{name}_equals_green_share": int(abs(share - ref) < 1e-9) if np.isfinite(ref) else -1,
            f"{name}_legs_vs_ship_years_pp": 100 * abs(chk["w_green_legs"] / chk["w_legs"] - share)
            if chk["w_legs"] > 0 else float("nan"),
            f"{name}_undrawable_w_pct": 100 * float(legs.loc[~drawable, "w"].sum() / legs["w"].sum())})
        print(f"  {name}: {len(s['nodes'])} nodes, share {100 * share:.2f}% "
              f"(green_share.csv {100 * ref:.2f}%), {len(agg):,} pairs")
    if not frames:
        sys.exit("no port set found")
    e = pd.concat(frames, ignore_index=True)
    base = e[e["set"] == frames[0]["set"].iloc[0]].sort_values("w", ascending=False)
    cum = base["w"].cumsum() / base["w"].sum()
    kept = base.loc[cum.shift(fill_value=0.0) < a.cover, ["a", "b"]]
    e = e.merge(kept, on=["a", "b"], how="inner")
    lab = np.asarray(labels, dtype=object)
    e["node_a"] = lab[e["a"].to_numpy()].astype(str)
    e["node_b"] = lab[e["b"].to_numpy()].astype(str)
    e = e[["set", "node_a", "node_b", "w", "w_green", "n_legs"]]
    e.to_csv(res / "edge_map.csv", index=False)
    facts.update({"pairs_kept": int(len(kept)), "cover_target_pct": 100 * a.cover,
                  "cover_pct": 100 * float(base.merge(kept, on=["a", "b"])["w"].sum() / base["w"].sum()),
                  "year": a.year})

    nd = (nodes.drop_duplicates("node_id")
          [["node_id", "node_name", "node_iso3", "node_lat", "node_lon"]].copy())
    nd["node_id"] = nd["node_id"].astype(str)
    used = set(e["node_a"]) | set(e["node_b"]) | set().union(*members.values())
    nd = nd[nd["node_id"].isin(used)]
    for name, m in members.items():
        nd[f"in_{name}"] = nd["node_id"].isin(m)
    nd.to_csv(res / "edge_map_nodes.csv", index=False)
    emit(res, "edge_map", {k: v for k, v in facts.items()
                           if not (isinstance(v, float) and not np.isfinite(v))})
    print(f"-> {res / 'edge_map.csv'} ({len(e):,} rows), edge_map_nodes.csv ({len(nd):,} nodes)")


if __name__ == "__main__":
    main()
