from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import emit

REFERENCE = 91.16
TARGETS = {2025: 0.02, 2030: 0.06, 2035: 0.145, 2040: 0.31, 2045: 0.62, 2050: 0.80}
PENALTY_EUR_PER_T_VLSFOE = 2400.0
RFNBO_REWARD, RFNBO_REWARD_UNTIL = 2.0, 2033
I_FOSSIL = REFERENCE
FUELS = {"e-methanol": (5.0, True), "bio-methanol": (14.0, False)}


def target(year: int) -> float:
    return REFERENCE * (1 - TARGETS[year])


def reward(fuel: str, year: int) -> float:
    return RFNBO_REWARD if FUELS[fuel][1] and year <= RFNBO_REWARD_UNTIL else 1.0


def intensity(g: float, fuel: str, year: int) -> float:
    im = FUELS[fuel][0]
    r = reward(fuel, year)
    return ((1 - g) * I_FOSSIL + g * im) / ((1 - g) + r * g)


def penalty(i: float, t: float) -> float:
    return PENALTY_EUR_PER_T_VLSFOE * (1 - t / i) if i > t else 0.0


def latest_target_met(g: float, fuel: str) -> int:
    met = [y for y in sorted(TARGETS) if intensity(g, fuel, y) <= target(y)]
    return max(met) if met else 0


def rows_for(g: float, meta: dict) -> list[dict]:
    out = []
    for fuel in FUELS:
        row = {**meta, "fuel": fuel, "green_share": g,
               "latest_target_met": latest_target_met(g, fuel)}
        for y in TARGETS:
            i = intensity(g, fuel, y)
            row[f"intensity_{y}"] = i
            row[f"penalty_eur_per_t_{y}"] = penalty(i, target(y))
        out.append(row)
    return out


METRICS = {"whole": ("green_ub_w", ""), "partial": ("green_p_ub_w", "_p"),
           "partial_eu": ("green_eu_ub_w", "_eu"), "partial_eea": ("green_eea_ub_w", "_eea"),
           "fuel": ("green_f_ub_w", "_f"), "fuel_eu": ("green_eu_f_ub_w", "_eu_f"),
           "opt_eu": ("green_eu_o_ub_w", "_eu_o"), "opt_eea": ("green_eea_o_ub_w", "_eea_o")}


def scenarios(results: Path) -> list[dict]:
    rows = []
    g = pd.read_csv(results / "green_share.csv")
    g = g[(g["fuel"] == "methanol") & (g["basis"] == "operating") & (g["reserve"] == 0.0)
          & (g["vessel_group"].isin(["container", "all"]))]
    for metric, (col, _) in METRICS.items():
        if col not in g:
            continue
        for r in g.itertuples():
            v = getattr(r, col)
            if pd.notna(v):
                rows += rows_for(float(v), {"metric": metric, "network": r.label,
                                            "set_key": r.set_key, "year": int(r.year),
                                            "vessel_group": r.vessel_group,
                                            "tank_mult": float(r.tank_mult)})
    p = results / "methanol_dated.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["variant"] == "all") & d["vessel_group"].isin(["container", "all"])]
        for metric, (_, infix) in METRICS.items():
            for r in d.itertuples():
                for mult, col in ((1.0, f"green_x1{infix}_ub_w"), (2.0, f"green_x2{infix}_ub_w")):
                    v = getattr(r, col, np.nan)
                    if pd.notna(v):
                        rows += rows_for(float(v), {"metric": metric,
                                                    "network": f"methanol network {r.year} ({r.bound})",
                                                    "set_key": f"methanol_dated|{r.bound}",
                                                    "year": int(r.year),
                                                    "vessel_group": r.vessel_group,
                                                    "tank_mult": mult})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    a = ap.parse_args(argv)
    res = a.results.expanduser().resolve()
    t = pd.DataFrame(scenarios(res))
    t.to_csv(res / "fueleu_network.csv", index=False)
    show = t[(t["vessel_group"] == "container")]
    print(show[["metric", "network", "year", "tank_mult", "fuel", "green_share",
                "latest_target_met", "penalty_eur_per_t_2040"]].round(3).to_string(index=False))

    f = {}
    def pick(network, year, mult, fuel, metric="whole"):
        r = t[(t["network"] == network) & (t["year"] == year) & (t["tank_mult"] == mult)
              & (t["fuel"] == fuel) & (t["vessel_group"] == "container")
              & (t["metric"] == metric)]
        return r.iloc[0] if len(r) else None
    for net, year, tag in (("Yap hubs (4)", 2024, "yap4"), ("20 chosen for continuity", 2024, "greedy"),
                           ("methanol network 2025 (upper)", 2025, "methdated_upper"),
                           ("methanol network 2025 (lower)", 2025, "methdated_lower")):
        for mult, mtag in ((1.0, "x1"), (2.0, "x2")):
            for fuel, ftag in (("e-methanol", "emeoh"), ("bio-methanol", "biomeoh")):
                for metric, pre in (("whole", ""), ("partial", "p_"), ("partial_eu", "eu_"),
                                    ("partial_eea", "eea_"), ("fuel", "f_"),
                                    ("fuel_eu", "feu_"), ("opt_eu", "oeu_"), ("opt_eea", "oeea_")):
                    r = pick(net, year, mult, fuel, metric)
                    if r is None:
                        continue
                    f[f"{pre}{tag}_{mtag}_{ftag}_latest_target"] = int(r["latest_target_met"])
                    f[f"{pre}{tag}_{mtag}_{ftag}_green_pct"] = 100 * float(r["green_share"])
                    for y in (2040, 2045, 2050):
                        f[f"{pre}{tag}_{mtag}_{ftag}_penalty_{y}_eur"] = float(
                            r[f"penalty_eur_per_t_{y}"])
                    f[f"{pre}{tag}_{mtag}_{ftag}_intensity_2040"] = float(r["intensity_2040"])
    emit(res, "fueleu", {k: v for k, v in f.items() if np.isfinite(v)})


if __name__ == "__main__":
    main()
