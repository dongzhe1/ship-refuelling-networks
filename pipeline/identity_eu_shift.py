from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
import emissions
from analyze import selected_key
from common import require
from select_ports import load_inputs
from supplement import (DUALFUEL_KEY, endurance_lookup, fuel_intensity, full_rows,
                        operating_endurance, scen_tag, scope_weights, set_tables,
                        tank_tonnes)

EVAL, DATED_YEAR, R = 2024, 2025, 7000
SCEN = ("methanol", "operating", 1.0, 0.0)
TAG = scen_tag(*SCEN)
METRICS = {"world_partial": (f"pub_{TAG}", "w_total"),
           "eu_sailing_order": (f"euub_{TAG}", "w_eu"),
           "eu_counted_first": (f"euoub_{TAG}", "w_eu")}
KEEP = ["imo", "year", "w_total", "w_eu", "operating_methanol_nm", "req_lb"] + \
       [c for c, _ in METRICS.values()]
RUNS = ("first", "every")


def network_defs(out: Path, EVAL: int, DATED_YEAR: int) -> dict[str, tuple[int, list[str]]]:
    ps = pd.concat([pd.read_csv(out / f, dtype={"node_id": str})
                    for f in ("port_sets.csv", "port_sets_dualfuel.csv") if (out / f).exists()],
                   ignore_index=True)

    def ranked(key, n):
        g = ps[ps["set_key"] == key].sort_values("rank")
        if g.empty:
            sys.exit(f"{out}: no {key} in the port sets")
        return g.loc[g["rank"] <= n, "node_id"].astype(str).tolist() if n else \
            g["node_id"].astype(str).tolist()
    d = pd.read_csv(out / "results" / "methanol_dated_sets.csv", keep_default_na=False)
    d = d[(d["variant"] == "all") & (d["year"] == DATED_YEAR)].set_index("bound")
    return {
        "Yap hubs (4)": (EVAL, ranked("external:yap4|-1|0|all", 0)),
        "methanol network, lower": (DATED_YEAR, d.loc["lower", "nodes"].split(";")),
        "methanol network, upper": (DATED_YEAR, d.loc["upper", "nodes"].split(";")),
        "10 chosen for dual-fuel use": (EVAL, ranked(DUALFUEL_KEY.format(year=EVAL, R=R), 10)),
        "10 chosen for continuity": (EVAL, ranked(selected_key("greedy", EVAL, R, "all"), 10)),
        "20 chosen for continuity": (EVAL, ranked(selected_key("greedy", EVAL, R, "all"), 20)),
    }


SELECTED = ("10 chosen for dual-fuel use", "10 chosen for continuity", "20 chosen for continuity")


def map_nodes(ids: list[str], src: Path, dst: Path) -> list[str]:
    a = pd.read_csv(src / "nodes.csv", dtype={"anchorage_id": str, "node_id": str})
    b = pd.read_csv(dst / "nodes.csv", dtype={"anchorage_id": str, "node_id": str})
    anch = a.loc[a["node_id"].isin(ids), "anchorage_id"]
    return sorted(set(b.loc[b["anchorage_id"].isin(anch), "node_id"]))


def ship_years(out: Path, nets: dict, calibration: Path | None) -> tuple[pd.DataFrame, set]:
    require(out, "stops.csv.gz", "nodes.csv", "vessels.csv", stage="the main run")
    stops, nodes, labels = load_inputs(out, extra=("sea_hours",))
    vessels = pd.read_csv(out / "vessels.csv", dtype={"imo": str})
    group_of = vessels.set_index("imo")["group"]
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, calibration)
    stops["leg_fuel_t"] = stops["leg_co2_t"] / emissions.CO2_PER_FUEL_T
    endur_vy, endur_v = endurance_lookup(operating_endurance(vessels, fuel_intensity(stops)),
                                         vessels)
    prep = chains.prepare(stops, labels, weight_col="leg_co2_t")
    scopes = {"eu": scope_weights(prep, stops, nodes)["eu"]}
    imos = set(stops["imo"].unique())
    del stops
    gc.collect()
    tanks = tank_tonnes(vessels)
    frames = []
    for label, (year, ids) in nets.items():
        t = set_tables(prep, chains.refuel_mask(prep, ids), endur_vy, endur_v, ranges=(R,),
                       scenarios=[SCEN], scopes=scopes, tanks=tanks)
        fy = full_rows(t, year, group_of)
        fy = fy[(fy["vessel_group"] == "container") & fy["operating_methanol_nm"].notna()]
        frames.append(fy[KEEP].assign(network=label, n_nodes=len(ids)))
        print(f"  {label}: {len(ids)} nodes, {len(fy):,} container ship-years with known tanks")
        del t
        gc.collect()
    del prep
    gc.collect()
    return pd.concat(frames, ignore_index=True), imos


def share(df: pd.DataFrame, num: str, den: str) -> float:
    d = df[den].sum()
    return float(df[num].sum() / d) if d > 0 else np.nan


def decompose(a: pd.DataFrame, b: pd.DataFrame, imos_first: set) -> list[dict]:
    key = ["imo", "year"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    ka = set(map(tuple, a[key].to_numpy()))
    kb = set(map(tuple, b[key].to_numpy()))
    a_only = a[[k not in kb for k in map(tuple, a[key].to_numpy())]]
    b_only = b[[k not in ka for k in map(tuple, b[key].to_numpy())]]
    b_new_ship = b_only[~b_only["imo"].isin(imos_first)]
    rows = []
    for metric, (num, den) in METRICS.items():
        S_a, S_b = share(a, num, den), share(b, num, den)
        Sm_a = share(m, f"{num}_a", f"{den}_a")
        Sm_b = share(m, f"{num}_b", f"{den}_b")
        wa, wb = m[f"{den}_a"].to_numpy(float), m[f"{den}_b"].to_numpy(float)
        sa = np.divide(m[f"{num}_a"].to_numpy(float), wa, out=np.zeros_like(wa), where=wa > 0)
        sb = np.divide(m[f"{num}_b"].to_numpy(float), wb, out=np.zeros_like(wb), where=wb > 0)
        Wa, Wb = wa.sum(), wb.sum()
        itin_share = float((wa / Wa * (sb - sa)).sum()) if Wa > 0 else np.nan
        itin_weight = float(((wb / Wb - wa / Wa) * sb).sum()) if Wa > 0 and Wb > 0 else np.nan
        rows.append({
            "metric": metric,
            "first_pct": 100 * S_a, "every_pct": 100 * S_b, "change_pp": 100 * (S_b - S_a),
            "first_matched_pct": 100 * Sm_a, "every_matched_pct": 100 * Sm_b,
            "drop_pp": 100 * (Sm_a - S_a), "itin_pp": 100 * (Sm_b - Sm_a),
            "itin_share_pp": 100 * itin_share, "itin_weight_pp": 100 * itin_weight,
            "add_pp": 100 * (S_b - Sm_b),
            "first_only_pct": 100 * share(a_only, num, den),
            "every_only_pct": 100 * share(b_only, num, den),
            "every_only_new_ship_pct": 100 * share(b_new_ship, num, den),
            "n_first": len(a), "n_every": len(b), "n_matched": len(m),
            "n_first_only": len(a_only), "n_every_only": len(b_only),
            "n_every_only_new_ship": len(b_new_ship),
            "den_first_only_share": float(a_only[den].sum() / a[den].sum()) if a[den].sum() > 0 else np.nan,
            "den_every_only_share": float(b_only[den].sum() / b[den].sum()) if b[den].sum() > 0 else np.nan,
            "den_every_only_new_ship_share": (float(b_new_ship[den].sum() / b[den].sum())
                                              if b[den].sum() > 0 else np.nan),
            "den_matched_ratio": float(Wb / Wa) if Wa > 0 else np.nan,
            "endurance_matched_ratio_p50": float(
                (m["operating_methanol_nm_b"] / m["operating_methanol_nm_a"]).median()),
        })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("first_out", type=Path)
    ap.add_argument("every_out", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--eval", dest="eval_", type=int, default=EVAL)
    ap.add_argument("--dated-year", type=int, default=DATED_YEAR)
    a = ap.parse_args(argv)
    outs = {"first": a.first_out.expanduser().resolve(), "every": a.every_out.expanduser().resolve()}
    dest = a.dest.expanduser().resolve()
    if dest in outs.values():
        sys.exit("dest must differ from both out directories (read only)")
    dest.mkdir(parents=True, exist_ok=True)

    own = {r: network_defs(outs[r], a.eval_, a.dated_year) for r in RUNS}
    nets = {r: dict(own[r]) for r in RUNS}
    for r, other in (("first", "every"), ("every", "first")):
        for label in SELECTED:
            year, ids = own[other][label]
            nets[r][f"{label} [{other}'s selection]"] = (year, map_nodes(ids, outs[other], outs[r]))

    sy, imos = {}, {}
    for r in RUNS:
        print(f"--- {r} identity run: {outs[r]}")
        sy[r], imos[r] = ship_years(outs[r], nets[r], a.calibration)
        sy[r].to_csv(dest / f"ship_years_{r}.csv.gz", index=False, compression="gzip")

    rows = []
    pairs = [(lab, lab, lab) for lab in own["first"]]
    for label in SELECTED:
        pairs.append((f"{label}, first run's nodes", label, f"{label} [first's selection]"))
        pairs.append((f"{label}, every run's nodes", f"{label} [every's selection]", label))
    for name, la, lb in pairs:
        A = sy["first"][sy["first"]["network"] == la]
        B = sy["every"][sy["every"]["network"] == lb]
        for row in decompose(A, B, imos["first"]):
            rows.append({"network": name, "n_nodes_first": int(A["n_nodes"].iloc[0]) if len(A) else 0,
                         "n_nodes_every": int(B["n_nodes"].iloc[0]) if len(B) else 0, **row})
    t = pd.DataFrame(rows)
    t.to_csv(dest / "identity_eu_shift.csv", index=False)
    facts = {f"{r['network']}|{r['metric']}|{c}": float(r[c]) for r in rows
             for c in ("first_pct", "every_pct", "change_pp", "drop_pp", "itin_pp", "add_pp")}
    (dest / "facts_identity_eu_shift.json").write_text(json.dumps(facts, indent=1) + "\n")
    show = t[t["metric"] != "eu_sailing_order"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(show[["network", "metric", "first_pct", "every_pct", "change_pp", "drop_pp", "itin_pp",
                    "itin_share_pp", "itin_weight_pp", "add_pp", "n_first", "n_every",
                    "n_matched", "den_every_only_share"]].round(2).to_string(index=False))
    print(f"-> {dest}/identity_eu_shift.csv")


if __name__ == "__main__":
    main()
