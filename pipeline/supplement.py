from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
import detour as dt
import emissions
from analyze import (EVAL, FUELS, GROUPS, JACCARD_BINS, N, RANGE, TRAIN, selected_key,
                     vessel_endurance)
from common import require
from evaluate import BASELINE_KEY, build_sets
from facts import emit
from select_ports import load_inputs

S2_RANGES = (5000, 7000, 10000, 20000)
SCEN_FUELS = ("methanol", "ammonia")
BASES = ("service", "operating")
MULTS = (1.0, 1.5, 2.0)
RESERVES = (0.0, 0.10)
BUNKER_N = 10

S1_BOTH_IN_MAIN_PP = 10.0
S2_ROBUST_FRACTION = 0.5
S3_CLAIM_FRACTION = 0.5
S4_ROBUST_FRACTION = 0.5
S5_EFFECT_PP = 20.0
S5_BOOT, S5_SEED = 1000, 20260928
S5_GREEN_CUT = 0.5

EU_FUELEU = frozenset("AUT BEL BGR HRV CYP CZE DNK EST FIN FRA DEU GRC HUN IRL ITA LVA "
                      "LTU LUX MLT NLD POL PRT ROU SVK SVN ESP SWE "
                      "GLP MTQ GUF REU MYT MAF".split())
SCOPES = {"eu": EU_FUELEU, "eea": EU_FUELEU | {"NOR", "ISL"}}
TRADEOFF_NS = (5, 10, 20, 50)
TRADEOFF_FAMILIES = ("all", "container")
TEU_M3 = 33.2
S8_FIRST_BY, S8_LAST_FROM = "01-31", "12-01"
S8_ROBUST_PP = 5.0
DETOUR_BUDGETS = (0.01, 0.03)
S10_CONT_PP, S10_GAIN_PP = 10.0, 10.0
S11_SHARED_MAX, S11_GAIN_PP = 15, 2.0
DUALFUEL_KEY = "dualfuel|{year}|{R}|all"


def scen_tag(fuel, basis, mult, reserve) -> str:
    return f"{fuel}_{basis}_x{mult:g}_r{int(round(100 * reserve))}"


SCENARIOS = [(f, b, m, r) for f in SCEN_FUELS for b in BASES for m in MULTS for r in RESERVES]
GENEROUS = ("methanol", "operating", 2.0, 0.0)
SCOPED = [(f, b, m, r) for f, b, m, r in SCENARIOS if f == "methanol" and r == 0.0]
FUEL_BASED = [(f, b, m, r) for f, b, m, r in SCOPED if b == "operating"]
TRADEOFF_SCEN = [("methanol", "operating", m, 0.0) for m in MULTS]


def key_sets(train=TRAIN, eval_=EVAL, R=RANGE, n=N) -> list[tuple[str, int, str]]:
    return [(BASELINE_KEY, -1, "all nodes"),
            ("external:yap4|-1|0|all", -1, "Yap hubs (4)"),
            ("external:yap8|-1|0|all", -1, "Yap hubs (8)"),
            ("external:bunker10|-1|0|all", -1, "top-10 bunker ports"),
            (selected_key("volume", eval_, 0, "all"), BUNKER_N, f"top {BUNKER_N} by calls"),
            (selected_key("greedy", eval_, R, "all"), BUNKER_N, f"{BUNKER_N} chosen for continuity"),
            (selected_key("volume", eval_, 0, "all"), n, f"top {n} by calls"),
            (selected_key("greedy", eval_, R, "all"), n, f"{n} chosen for continuity"),
            (selected_key("greedy", train, R, "all"), n, f"{n} chosen for continuity on {train}")]


def fuel_intensity(stops: pd.DataFrame) -> pd.DataFrame:
    ok = stops["leg_ok"].to_numpy() & (stops["leg_fuel_t"].to_numpy() > 0)
    s = stops.loc[ok, ["imo", "visit_start", "leg_fuel_t", "leg_nm"]]
    g = s.assign(year=s["visit_start"].dt.year).groupby(["imo", "year"])
    t = g.agg(fuel_t=("leg_fuel_t", "sum"), dist_nm=("leg_nm", "sum")).reset_index()
    t["t_per_nm"] = t["fuel_t"] / t["dist_nm"].where(t["dist_nm"] > 0)
    return t


def operating_endurance(vessels: pd.DataFrame, intensity: pd.DataFrame) -> pd.DataFrame:
    v = vessel_endurance(vessels)
    tank_t = pd.to_numeric(v["fuel_capacity_m3"], errors="coerce") * emissions.TANK_USABLE \
        * emissions.HFO_DENSITY_T_PER_M3
    tanks = pd.DataFrame({"imo": v["imo"], "tank_t": tank_t.to_numpy(), "group": v["group"],
                          **{f"service_{f}_nm": v[f"endurance_{f}_nm"] for f in FUELS}})
    e = intensity.merge(tanks, on="imo", how="left")
    hfo = e["tank_t"] / e["t_per_nm"]
    e["source"] = np.where(hfo.notna(), "own_intensity", "service_fallback")
    for f in FUELS:
        e[f"operating_{f}_nm"] = (hfo * emissions.volumetric_ratio(f)).fillna(e[f"service_{f}_nm"])
    return e


def endurance_lookup(e: pd.DataFrame, vessels: pd.DataFrame) -> pd.DataFrame:
    v = vessel_endurance(vessels)
    service = pd.DataFrame({"imo": v["imo"], **{f"service_{f}_nm": v[f"endurance_{f}_nm"]
                                                 for f in FUELS}})
    cols = ["imo", "year"] + [f"{b}_{f}_nm" for b in BASES for f in FUELS]
    return e[cols], service


def endurance_operating_summary(e: pd.DataFrame, year: int) -> pd.DataFrame:
    rows = []
    ey = e[e["year"] == year]
    for grp in GROUPS:
        p = ey if grp == "all" else ey[ey["group"] == grp]
        p = p.dropna(subset=["operating_hfo_nm", "service_hfo_nm"])
        if p.empty:
            continue
        row = {"year": year, "vessel_group": grp, "n_vessel_years": len(p),
               "share_own_intensity": float((p["source"] == "own_intensity").mean()),
               "ratio_operating_to_service_p50": float(
                   (p["operating_hfo_nm"] / p["service_hfo_nm"]).median())}
        for f in FUELS:
            row[f"operating_{f}_p50"] = float(p[f"operating_{f}_nm"].median())
            row[f"service_{f}_p50"] = float(p[f"service_{f}_nm"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def scope_weights(prep: chains.Prepared, stops: pd.DataFrame, nodes: pd.DataFrame
                  ) -> dict[str, np.ndarray]:
    iso = stops["anchorage_id"].map(nodes.drop_duplicates("anchorage_id")
                                    .set_index("anchorage_id")["iso3"]).fillna("")
    iso = iso.loc[prep.src].to_numpy().astype(str)
    out = {}
    for name, members in SCOPES.items():
        e = np.isin(iso, sorted(members)).astype(float)
        w = 0.5 * (np.r_[0.0, e[:-1]] + e)
        w[prep.new_run] = 0.0
        out[name] = w
    return out


def tank_tonnes(vessels: pd.DataFrame) -> pd.Series:
    t = pd.to_numeric(vessels["fuel_capacity_m3"], errors="coerce") \
        * emissions.TANK_USABLE * emissions.HFO_DENSITY_T_PER_M3
    return pd.Series(t.to_numpy(), index=vessels["imo"].astype(str)).groupby(level=0).first()


def vessel_span(prep: chains.Prepared) -> pd.DataFrame:
    if prep.n == 0:
        return pd.DataFrame(columns=["first", "last"])
    first = np.flatnonzero(prep.new_vessel)
    last = np.r_[first[1:] - 1, prep.n - 1]
    return pd.DataFrame({"first": prep.time[first], "last": prep.time[last]},
                        index=prep.imo[first])


def whole_year(t: pd.DataFrame, span: pd.DataFrame) -> np.ndarray:
    f = t["imo"].map(span["first"])
    l = t["imo"].map(span["last"])
    y = t["year"].astype(str)
    return ((f <= pd.to_datetime(y + "-" + S8_FIRST_BY + " 23:59:59"))
            & (l >= pd.to_datetime(y + "-" + S8_LAST_FROM))).to_numpy()


def set_tables(prep: chains.Prepared, refuel: np.ndarray, endur_vy: pd.DataFrame,
               endur_v: pd.DataFrame, ranges=S2_RANGES, scenarios=None,
               scopes: dict | None = None, strand_R: int | None = None,
               tanks: pd.Series | None = None) -> pd.DataFrame:
    scenarios = SCENARIOS if scenarios is None else scenarios
    seg = chains.segments(prep, refuel)
    sy = chains.ship_years(prep, seg, refuel, ranges)
    out = sy.set_index(["vessel", "year"])
    key_out = out.index.get_level_values(0).to_numpy() * 10000 + out.index.get_level_values(1).to_numpy()
    pos = np.searchsorted(key_out, prep.vessel * 10000 + prep.year)
    d0, anchored = chains.leg_start(prep, refuel)
    nrow = len(out)

    def per_row(v):
        return np.bincount(pos, weights=v, minlength=nrow)

    right = seg["right"].to_numpy()
    s = seg.assign(year=prep.year[right] if len(seg) else np.zeros(0, int),
                   imo=prep.imo[right] if len(seg) else np.zeros(0, str),
                   censored=(seg["kind"] != chains.COMPLETE).to_numpy())
    new = {}
    for R in ranges:
        add = (s[s["censored"] & (s["length_nm"] <= R)]
               .groupby(["vessel", "year"])[["weight", "length_nm"]].sum()
               .reindex(out.index).fillna(0.0))
        lb_w = out[f"covered_w_{int(R)}"] if f"covered_w_{int(R)}" in out else out[f"covered_{int(R)}"]
        new[f"lb_w_{int(R)}"] = lb_w
        new[f"ub_w_{int(R)}"] = lb_w + add["weight"]
        new[f"lb_d_{int(R)}"] = out[f"covered_{int(R)}"]
        new[f"ub_d_{int(R)}"] = out[f"covered_{int(R)}"] + add["length_nm"]
    def attach(frame):
        f = frame.merge(endur_vy, on=["imo", "year"], how="left")
        f = f.merge(endur_v, on="imo", how="left", suffixes=("", "_v"))
        for fu in FUELS:
            f[f"service_{fu}_nm"] = f[f"service_{fu}_nm"].fillna(f[f"service_{fu}_nm_v"])
            f[f"operating_{fu}_nm"] = f[f"operating_{fu}_nm"].fillna(f[f"service_{fu}_nm"])
        return f
    segs = attach(s[["vessel", "year", "imo", "length_nm", "weight", "censored"]].reset_index(drop=True))
    oy = attach(out.reset_index()[["vessel", "year", "imo", "req_lb"]])
    oy = oy.set_index(["vessel", "year"])
    if scopes:
        for sc, w in scopes.items():
            new[f"w_{sc}"] = pd.Series(per_row(prep.leg_w * w), index=out.index)
    for fuel, basis, mult, res in scenarios:
        tag = scen_tag(fuel, basis, mult, res)
        Es = oy[f"{basis}_{fuel}_nm"].reindex(out.index).to_numpy()[pos] * mult * (1 - res)
        f_lb = chains.fraction_within(prep, d0, anchored, Es, anchored_only=True)
        f_ub = chains.fraction_within(prep, d0, anchored, Es, anchored_only=False)
        new[f"plb_{tag}"] = pd.Series(per_row(f_lb * prep.leg_w), index=out.index)
        new[f"pub_{tag}"] = pd.Series(per_row(f_ub * prep.leg_w), index=out.index)
        if scopes and (fuel, basis, mult, res) in SCOPED:
            for sc, w in scopes.items():
                new[f"{sc}lb_{tag}"] = pd.Series(per_row(f_lb * prep.leg_w * w), index=out.index)
                new[f"{sc}ub_{tag}"] = pd.Series(per_row(f_ub * prep.leg_w * w), index=out.index)
        E = segs[f"{basis}_{fuel}_nm"].to_numpy() * mult * (1 - res)
        ok = segs["length_nm"].to_numpy() <= E
        w = segs["weight"].to_numpy()
        g = pd.DataFrame({"vessel": segs["vessel"], "year": segs["year"],
                          "lb": np.where(ok & ~segs["censored"].to_numpy(), w, 0.0),
                          "cz": np.where(ok & segs["censored"].to_numpy(), w, 0.0)}
                         ).groupby(["vessel", "year"])[["lb", "cz"]].sum().reindex(out.index).fillna(0.0)
        new[f"glb_{tag}"] = g["lb"]
        new[f"gub_{tag}"] = g["lb"] + g["cz"]
        Ey = oy[f"{basis}_{fuel}_nm"].reindex(out.index).to_numpy() * mult * (1 - res)
        new[f"fits_{tag}"] = np.where(np.isnan(Ey), np.nan,
                                      (out["req_lb"].to_numpy() <= Ey).astype(float))
    if tanks is not None and prep.weighted:
        w0, anc_w = chains.leg_start(prep, refuel, by="weight")
        tank_stop = pd.Series(prep.imo).map(tanks).to_numpy(float)
        per_t = emissions.volumetric_ratio("methanol") * emissions.CO2_PER_FUEL_T
        for fuel, basis, mult, res in (x for x in scenarios if x in FUEL_BASED):
            tag = scen_tag(fuel, basis, mult, res)
            C = tank_stop * per_t * mult * (1 - res)
            f_lb = chains.fraction_within(prep, w0, anc_w, C, anchored_only=True, by="weight")
            f_ub = chains.fraction_within(prep, w0, anc_w, C, anchored_only=False, by="weight")
            new[f"flb_{tag}"] = pd.Series(per_row(f_lb * prep.leg_w), index=out.index)
            new[f"fub_{tag}"] = pd.Series(per_row(f_ub * prep.leg_w), index=out.index)
            for sc, w in (scopes or {}).items():
                new[f"{sc}flb_{tag}"] = pd.Series(per_row(f_lb * prep.leg_w * w), index=out.index)
                new[f"{sc}fub_{tag}"] = pd.Series(per_row(f_ub * prep.leg_w * w), index=out.index)
                o_lb = chains.allocate_by_priority(prep, refuel, C, w, anchored_only=True)
                o_ub = chains.allocate_by_priority(prep, refuel, C, w, anchored_only=False)
                new[f"{sc}olb_{tag}"] = pd.Series(per_row(o_lb * prep.leg_w * w), index=out.index)
                new[f"{sc}oub_{tag}"] = pd.Series(per_row(o_ub * prep.leg_w * w), index=out.index)
    if strand_R is not None:
        f_R = chains.fraction_within(prep, d0, anchored, float(strand_R), anchored_only=True)
        new[f"plb_R{int(strand_R)}"] = pd.Series(per_row(f_R * prep.leg_w), index=out.index)
    for basis in BASES:
        for fu in FUELS:
            new[f"{basis}_{fu}_nm"] = oy[f"{basis}_{fu}_nm"].reindex(out.index)
    out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)
    return out.reset_index()


def full_rows(t: pd.DataFrame, year: int, group_of) -> pd.DataFrame:
    t = t[(t["year"] == year) & (t["n_breaks"] == 0) & t["req_lb"].notna()].copy()
    t["vessel_group"] = t["imo"].map(group_of).fillna("unknown")
    return t


def _groups(t):
    for grp in GROUPS:
        p = t if grp == "all" else t[t["vessel_group"] == grp]
        if len(p):
            yield grp, p


def s1_rows(t, key, label, year):
    rows = []
    for grp, p in _groups(t.dropna(subset=["service_hfo_nm", "operating_hfo_nm"])):
        w = p["w_total"].to_numpy(float)
        row = {"set_key": key, "label": label, "year": year, "vessel_group": grp, "n": len(p)}
        for basis in BASES:
            for f in FUELS:
                ok = (p["req_lb"] <= p[f"{basis}_{f}_nm"]).to_numpy()
                row[f"{basis}_fits_{f}_w"] = float(w[ok].sum() / w.sum()) if w.sum() > 0 else np.nan
        rows.append(row)
    return rows


def s2_rows(t, key, label, year, ranges=S2_RANGES):
    rows = []
    for grp, p in _groups(t):
        W, D = p["w_total"].sum(), p["dist_nm"].sum()
        for R in ranges:
            lb = p[f"lb_w_{int(R)}"] / p["w_total"].where(p["w_total"] > 0)
            ub = p[f"ub_w_{int(R)}"] / p["w_total"].where(p["w_total"] > 0)
            rows.append({
                "set_key": key, "label": label, "year": year, "vessel_group": grp,
                "range_nm": int(R), "n_full": len(p),
                "lb_w": float(p[f"lb_w_{int(R)}"].sum() / W) if W > 0 else np.nan,
                "ub_w": float(p[f"ub_w_{int(R)}"].sum() / W) if W > 0 else np.nan,
                "lb_d": float(p[f"lb_d_{int(R)}"].sum() / D) if D > 0 else np.nan,
                "ub_d": float(p[f"ub_d_{int(R)}"].sum() / D) if D > 0 else np.nan,
                "lb_p25": float(lb.quantile(.25)), "lb_p50": float(lb.quantile(.5)),
                "lb_p75": float(lb.quantile(.75)),
                "lb_ge50": float((lb >= 0.5).mean()), "lb_ge90": float((lb >= 0.9).mean()),
                "ub_ge50": float((ub >= 0.5).mean()), "ub_ge90": float((ub >= 0.9).mean())})
    return rows


def green_cols(k: pd.DataFrame, tag: str) -> dict:
    Wk = k["w_total"].sum()
    r = lambda col: float(k[col].sum() / Wk) if Wk > 0 and col in k else np.nan
    out = {"green_lb_w": r(f"glb_{tag}"), "green_ub_w": r(f"gub_{tag}"),
           "green_p_lb_w": r(f"plb_{tag}"), "green_p_ub_w": r(f"pub_{tag}"),
           "green_f_lb_w": r(f"flb_{tag}"), "green_f_ub_w": r(f"fub_{tag}")}
    for sc in SCOPES:
        Ws = k[f"w_{sc}"].sum() if f"w_{sc}" in k else 0.0
        for kind, pre in (("", sc), ("_f", f"{sc}f"), ("_o", f"{sc}o")):
            for b in ("lb", "ub"):
                col = f"{pre}{b}_{tag}"
                out[f"green_{sc}{kind}_{b}_w"] = (float(k[col].sum() / Ws)
                                                  if Ws > 0 and col in k else np.nan)
    return out


def s34_rows(t, key, label, year, scenarios=None):
    tank, green = [], []
    for grp, p in _groups(t):
        w = p["w_total"].to_numpy(float)
        for fuel, basis, mult, res in (SCENARIOS if scenarios is None else scenarios):
            tag = scen_tag(fuel, basis, mult, res)
            base = {"set_key": key, "label": label, "year": year, "vessel_group": grp,
                    "fuel": fuel, "basis": basis, "tank_mult": mult, "reserve": res}
            fits = p[f"fits_{tag}"].to_numpy(float)
            known = ~np.isnan(fits)
            tank.append({**base, "n": int(known.sum()),
                         "fits_w": float((w[known] * fits[known]).sum() / w[known].sum())
                         if w[known].sum() > 0 else np.nan,
                         "fits": float(np.nanmean(fits)) if known.any() else np.nan})
            k = p[p[f"{basis}_{fuel}_nm"].notna()]
            Wk = k["w_total"].sum()
            frac = k[f"glb_{tag}"] / k["w_total"].where(k["w_total"] > 0)
            pfrac = k[f"plb_{tag}"] / k["w_total"].where(k["w_total"] > 0)
            green.append({**base, "n": len(k), "n_unknown_tank": int(len(p) - len(k)),
                          **green_cols(k, tag),
                          "touch_w": float(k.loc[k["touches"].astype(bool), "w_total"].sum() / Wk)
                          if Wk > 0 else np.nan,
                          "lb_p50": float(frac.median()), "lb_ge50": float((frac >= 0.5).mean()),
                          "p_lb_p50": float(pfrac.median()),
                          "p_lb_ge50": float((pfrac >= 0.5).mean())})
    return tank, green


def _rate(y, mask):
    m = mask.sum()
    return y[mask].mean() if m else np.nan


def stranding(t: pd.DataFrame, red: pd.DataFrame, group_of, train: int, eval_: int,
              R: int, boot: int = S5_BOOT, seed: int = S5_SEED) -> pd.DataFrame:
    full = t[(t["n_breaks"] == 0) & t["req_lb"].notna()]
    a = full[full["year"] == train].set_index("imo")
    b = full[full["year"] == eval_].set_index("imo")
    cols = ["req_lb", "w_total", f"lb_w_{int(R)}"]
    b2 = f"plb_R{int(R)}"
    cols += [b2] if b2 in full else []
    j = a[cols].join(b[cols], lsuffix="_t", rsuffix="_y", how="inner")
    rj = red[(red["train_year"] == train) & (red["year"] == eval_)].set_index("imo")["jaccard"]
    j = j.join(rj, how="inner")
    j["vessel_group"] = j.index.map(group_of).fillna("unknown")
    j["bin"] = np.digitize(j["jaccard"].to_numpy(), JACCARD_BINS[1:-1])
    j["feas_t"], j["feas_y"] = j["req_lb_t"] <= R, j["req_lb_y"] <= R
    gt = j[f"lb_w_{int(R)}_t"] / j["w_total_t"].where(j["w_total_t"] > 0)
    gy = j[f"lb_w_{int(R)}_y"] / j["w_total_y"].where(j["w_total_y"] > 0)
    j["green_t"], j["green_y"] = gt >= S5_GREEN_CUT, gy < S5_GREEN_CUT
    outcomes = [("A", "feas_t", "feas_y", True), ("B", "green_t", "green_y", False)]
    if f"{b2}_t" in j:
        pt = j[f"{b2}_t"] / j["w_total_t"].where(j["w_total_t"] > 0)
        py = j[f"{b2}_y"] / j["w_total_y"].where(j["w_total_y"] > 0)
        j["green2_t"], j["green2_y"] = pt >= S5_GREEN_CUT, py < S5_GREEN_CUT
        outcomes.append(("B2", "green2_t", "green2_y", False))
    rng = np.random.default_rng(seed)
    rng_b2 = np.random.default_rng(seed + 1)
    rows = []
    for grp in GROUPS:
        p = j if grp == "all" else j[j["vessel_group"] == grp]
        for outcome, pop_col, lost_col, negate in outcomes:
            lost = ~p[lost_col] if negate else p[lost_col]
            pop = p[pop_col].to_numpy()
            if pop.sum() < 2:
                continue
            y = lost.to_numpy()[pop].astype(float)
            bins = p["bin"].to_numpy()[pop]
            est = {"overall": y.mean(), "effect": _rate(y, bins == 0) - _rate(y, bins == 4)}
            for k in range(5):
                est[f"bin{k}"] = _rate(y, bins == k)
            draws = {k: [] for k in est}
            n = len(y)
            gen = rng_b2 if outcome == "B2" else rng
            for _ in range(boot):
                i = gen.integers(0, n, n)
                yy, bb = y[i], bins[i]
                draws["overall"].append(yy.mean())
                draws["effect"].append(_rate(yy, bb == 0) - _rate(yy, bb == 4))
                for k in range(5):
                    draws[f"bin{k}"].append(_rate(yy, bb == k))
            for k, v in est.items():
                d = np.array(draws[k], float)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lo, hi = np.nanquantile(d, 0.025), np.nanquantile(d, 0.975)
                rows.append({"vessel_group": grp, "outcome": outcome, "metric": k,
                             "estimate": float(v), "ci_lo": float(lo), "ci_hi": float(hi),
                             "n_population": int(n),
                             "n_in_bin": int((bins == int(k[3:])).sum()) if k.startswith("bin") else int(n)})
    return pd.DataFrame(rows)


def readings(own, seg, tank, green, strand, cov, eval_=EVAL, R=RANGE) -> dict:
    f = {}
    c = own[own["vessel_group"] == "container"]
    if len(c):
        d = 100 * (c["operating_fits_methanol_w"] - c["service_fits_methanol_w"])
        f["s1_max_methanol_change_pp"] = float(d.abs().max())
        f["s1_reading"] = ("both versions in the main text"
                           if f["s1_max_methanol_change_pp"] > S1_BOTH_IN_MAIN_PP
                           else "service speed in the main text, own intensity in the appendix")
        for _, r in c.iterrows():
            m, y = r["set_key"].split("|")[:2]
            tag = m.replace("external:", "") + ("" if y == "-1" else f"{y}n{int(r['n_ports'])}")
            f[f"s1_container_{tag}_methanol_service_pct"] = 100 * float(r["service_fits_methanol_w"])
            f[f"s1_container_{tag}_methanol_operating_pct"] = 100 * float(r["operating_fits_methanol_w"])
    yap = "external:yap4|-1|0|all"
    t = cov[(cov["set_key"] == yap) & (cov["vessel_group"] == "container")
            & (cov["eval_year"] == eval_) & (cov["range_nm"] == R)]
    touch = float(t["touch_share_w"].iloc[0]) if len(t) else np.nan
    f["yap4_container_touch_pct"] = 100 * touch
    y2 = seg[(seg["set_key"] == yap) & (seg["vessel_group"] == "container") & (seg["range_nm"] == R)]
    if len(y2) and touch == touch:
        ub = float(y2["ub_w"].iloc[0])
        f.update({"s2_yap4_container_lb_pct": 100 * float(y2["lb_w"].iloc[0]),
                  "s2_yap4_container_ub_pct": 100 * ub,
                  "s2_reading": "robust" if ub < S2_ROBUST_FRACTION * touch
                  else "state the contrast with the coverage bounds"})
    fuel, basis, mult, res = GENEROUS
    sel = lambda d: d[(d["set_key"] == yap) & (d["vessel_group"] == "container") & (d["fuel"] == fuel)
                      & (d["basis"] == basis) & (d["tank_mult"] == mult) & (d["reserve"] == res)]
    t3 = sel(tank)
    if len(t3):
        v = float(t3["fits_w"].iloc[0])
        f["s3_yap4_container_generous_fits_pct"] = 100 * v
        f["s3_reading"] = ("claim holds beyond same-volume tanks" if v < S3_CLAIM_FRACTION
                           else "restrict the claim to same-volume tanks")
    t4 = sel(green)
    if len(t4) and touch == touch:
        ub = float(t4["green_ub_w"].iloc[0])
        f["s4_yap4_container_generous_green_lb_pct"] = 100 * float(t4["green_lb_w"].iloc[0])
        f["s4_yap4_container_generous_green_ub_pct"] = 100 * ub
        f["s4_yap4_container_touch_known_tank_pct"] = 100 * float(t4["touch_w"].iloc[0]) \
            if "touch_w" in t4 else float("nan")
        f["s4_reading"] = ("robust" if ub < S4_ROBUST_FRACTION * touch
                           else "state the contrast with the green-share bounds")
        if "green_p_ub_w" in t4:
            pub = float(t4["green_p_ub_w"].iloc[0])
            f["s4p_yap4_container_generous_green_lb_pct"] = 100 * float(t4["green_p_lb_w"].iloc[0])
            f["s4p_yap4_container_generous_green_ub_pct"] = 100 * pub
            f["s4p_reading"] = ("robust" if pub < S4_ROBUST_FRACTION * touch
                                else "state the contrast with the green-share bounds")
    a = strand[(strand["vessel_group"] == "all") & (strand["outcome"] == "A")] \
        if len(strand) else strand
    if len(a):
        e = a[a["metric"] == "effect"].iloc[0]
        o = a[a["metric"] == "overall"].iloc[0]
        f.update({"s5_stranding_overall_pct": 100 * o["estimate"],
                  "s5_effect_pp": 100 * e["estimate"], "s5_effect_ci_lo_pp": 100 * e["ci_lo"],
                  "s5_effect_ci_hi_pp": 100 * e["ci_hi"]})
        f["s5_reading"] = ("not estimable (an overlap bin is empty)" if e["estimate"] != e["estimate"]
                           else "redeployment strands ships: supported"
                           if 100 * e["estimate"] >= S5_EFFECT_PP and e["ci_lo"] > 0
                           else "not supported")
    b2 = strand[(strand["vessel_group"] == "all") & (strand["outcome"] == "B2")] \
        if len(strand) else strand
    if len(b2):
        e = b2[b2["metric"] == "effect"].iloc[0]
        o = b2[b2["metric"] == "overall"].iloc[0]
        f.update({"s5b2_overall_pct": 100 * o["estimate"], "s5b2_effect_pp": 100 * e["estimate"],
                  "s5b2_effect_ci_lo_pp": 100 * e["ci_lo"], "s5b2_effect_ci_hi_pp": 100 * e["ci_hi"],
                  "s5b2_population": int(o["n_population"])})
    return {k: v for k, v in f.items() if not (isinstance(v, float) and v != v)}


def stop_endurance(prep: chains.Prepared, endur_vy, endur_v, col="operating_methanol_nm"):
    key = pd.DataFrame({"imo": prep.imo, "year": prep.year})
    u = key.drop_duplicates().merge(endur_vy[["imo", "year", col]], on=["imo", "year"], how="left")
    fb = col.replace("operating_", "service_")
    u = u.merge(endur_v[["imo", fb]], on="imo", how="left")
    u[col] = u[col].fillna(u[fb])
    return key.merge(u[["imo", "year", col]], on=["imo", "year"], how="left")[col].to_numpy(float)


def s10_rows(prep, refuel, det, E_stop, R, endur_vy, endur_v, group_of, eval_, meta,
             tanks=None) -> list[dict]:
    rows = []
    in_year = prep.year == eval_
    dist = float(prep.leg_nm[in_year].sum())
    tag = {m: scen_tag("methanol", "operating", m, 0.0) for m in (1.0, 2.0)}
    for delta in (0.0,) + DETOUR_BUDGETS:
        cells, info = {}, {}
        for m in (1.0, 2.0):
            r2, st = dt.insert_refuels(prep, refuel, E_stop * m, det, delta)
            t = set_tables(prep, r2, endur_vy, endur_v, ranges=(R,),
                           scenarios=[("methanol", "operating", m, 0.0)])
            cells[m] = full_rows(t, eval_, group_of)
            info[m] = st
        rR, stR = dt.insert_refuels(prep, refuel, float(R), det, delta)
        tR = full_rows(set_tables(prep, rR, endur_vy, endur_v, ranges=(R,), scenarios=[]),
                       eval_, group_of)
        for grp in ("all", "container"):
            row = {**meta, "budget": delta, "vessel_group": grp}
            for m, mt in ((1.0, "x1"), (2.0, "x2")):
                p = cells[m] if grp == "all" else cells[m][cells[m]["vessel_group"] == grp]
                k = p[p["operating_methanol_nm"].notna()]
                Wk = k["w_total"].sum()
                row[f"green_p_ub_{mt}_w"] = float(k[f"pub_{tag[m]}"].sum() / Wk) if Wk > 0 else np.nan
            q = tR if grp == "all" else tR[tR["vessel_group"] == grp]
            W = q["w_total"].sum()
            row["feasible_w"] = float(q.loc[q["req_lb"] <= R, "w_total"].sum() / W) if W > 0 else np.nan
            row["segments_with_detour_pct"] = (100 * info[1.0]["inserted"] / info[1.0]["segments"]
                                               if info[1.0]["segments"] else 0.0)
            ch = info[1.0]["chosen"]
            extra = float(det[ch][in_year[ch]].sum()) if len(ch) else 0.0
            row["extra_distance_pct"] = 100 * extra / dist if dist > 0 else 0.0
            rows.append(row)
    return rows


def s10_readings(d: pd.DataFrame) -> dict:
    f = {}
    y = d[(d["set_key"] == "external:yap4|-1|0|all") & (d["vessel_group"] == "container")] \
        .set_index("budget") if len(d) else d
    if len(y) and 0.0 in y.index and 0.03 in y.index:
        f["s10_yap4_feasible_pct"] = 100 * y.loc[0.0, "feasible_w"]
        f["s10_yap4_feasible_3pct"] = 100 * y.loc[0.03, "feasible_w"]
        f["s10_yap4_green_x1_pct"] = 100 * y.loc[0.0, "green_p_ub_x1_w"]
        f["s10_yap4_green_x1_3pct"] = 100 * y.loc[0.03, "green_p_ub_x1_w"]
        f["s10_yap4_green_x1_gain_pp"] = f["s10_yap4_green_x1_3pct"] - f["s10_yap4_green_x1_pct"]
        f["s10_reading"] = ("a few detours do not rescue the network: infrastructure binds"
                            if f["s10_yap4_feasible_3pct"] < S10_CONT_PP
                            and f["s10_yap4_green_x1_gain_pp"] < S10_GAIN_PP
                            else "small detours improve the network markedly")
    return f


def s11_rows(fy: pd.DataFrame, key, label, n, R) -> list[dict]:
    rows = []
    for grp, p in _groups(fy):
        if grp not in ("all", "container"):
            continue
        W = p["w_total"].sum()
        row = {"set_key": key, "label": label, "n_ports": n, "vessel_group": grp,
               "touch_w": float(p.loc[p["touches"].astype(bool), "w_total"].sum() / W) if W > 0 else np.nan,
               "feasible_w": float(p.loc[p["req_lb"] <= R, "w_total"].sum() / W) if W > 0 else np.nan}
        for m, mt in ((1.0, "x1"), (2.0, "x2")):
            tag = scen_tag("methanol", "operating", m, 0.0)
            g = green_cols(p[p["operating_methanol_nm"].notna()], tag)
            row[f"green_p_ub_{mt}_w"] = g["green_p_ub_w"]
            row[f"green_eu_o_ub_{mt}_w"] = g["green_eu_o_ub_w"]
        rows.append(row)
    return rows


def s11_overlap(built, years, R, ns=(10, 20)) -> pd.DataFrame:
    def nodes(key, n):
        s = next((x for x in built if x["set_key"] == key and x["n_ports"] == n), None)
        return set(s["nodes"]) if s else None
    rows = []
    for y in years:
        for n in ns:
            sets = {"volume": nodes(selected_key("volume", y, 0, "all"), n),
                    "greedy": nodes(selected_key("greedy", y, R, "all"), n),
                    "dualfuel": nodes(DUALFUEL_KEY.format(year=y, R=R), n)}
            for a_, b_ in (("greedy", "dualfuel"), ("volume", "dualfuel"), ("volume", "greedy")):
                if sets[a_] is None or sets[b_] is None:
                    continue
                rows.append({"train_year": y, "n_ports": n, "a": a_, "b": b_,
                             "shared": len(sets[a_] & sets[b_])})
    return pd.DataFrame(rows)


def s11_readings(rows: pd.DataFrame, overlap: pd.DataFrame, eval_=EVAL, R=RANGE, n=N) -> dict:
    f = {}
    if not len(rows) or not len(overlap):
        return f
    c = rows[(rows["vessel_group"] == "container")]
    old = c[(c["set_key"] == selected_key("greedy", eval_, R, "all")) & (c["n_ports"] == n)]
    new = c[(c["set_key"] == DUALFUEL_KEY.format(year=eval_, R=R)) & (c["n_ports"] == n)]
    o = overlap[(overlap["train_year"] == eval_) & (overlap["n_ports"] == n)
                & (overlap["a"] == "greedy") & (overlap["b"] == "dualfuel")]
    if len(old) and len(new) and len(o):
        shared = int(o["shared"].iloc[0])
        gain = 100 * float(new["green_p_ub_x1_w"].iloc[0] - old["green_p_ub_x1_w"].iloc[0])
        f.update({"s11_shared_nodes": shared, "s11_green_x1_gain_pp": gain,
                  "s11_old_green_x1_pct": 100 * float(old["green_p_ub_x1_w"].iloc[0]),
                  "s11_new_green_x1_pct": 100 * float(new["green_p_ub_x1_w"].iloc[0]),
                  "s11_old_feasible_pct": 100 * float(old["feasible_w"].iloc[0]),
                  "s11_new_feasible_pct": 100 * float(new["feasible_w"].iloc[0])})
        f["s11_reading"] = ("the corrected measure changes where to invest"
                            if shared <= S11_SHARED_MAX * n / 20 and gain >= S11_GAIN_PP
                            else "the corrected measure changes the numbers, not the ports")
    return f


def s7_rows(fy: pd.DataFrame, meta: dict) -> list[dict]:
    rows = []
    for grp, p in _groups(fy):
        if grp not in ("all", "container"):
            continue
        for fuel, basis, mult, res in TRADEOFF_SCEN:
            tag = scen_tag(fuel, basis, mult, res)
            k = p[p[f"{basis}_{fuel}_nm"].notna()]
            Wk = k["w_total"].sum()
            rows.append({**meta, "vessel_group": grp, "tank_mult": mult, "n": len(k),
                         "touch_w": float(k.loc[k["touches"].astype(bool), "w_total"].sum() / Wk)
                         if Wk > 0 else np.nan, **green_cols(k, tag)})
    return rows


def tank_space(fy: pd.DataFrame, vessels: pd.DataFrame) -> pd.DataFrame:
    v = vessels.set_index("imo")
    c = fy[(fy["vessel_group"] == "container") & fy["operating_methanol_nm"].notna()]
    cap = pd.to_numeric(c["imo"].map(v["fuel_capacity_m3"]), errors="coerce")
    teu = pd.to_numeric(c["imo"].map(v["teu"]), errors="coerce") if "teu" in v else cap * np.nan
    ok = cap.notna() & (teu > 0)
    rows = []
    for m in MULTS:
        extra = (m - 1) * cap[ok] / TEU_M3
        rows.append({"tank_mult": m, "n": int(ok.sum()),
                     "extra_teu_p50": float(extra.median()) if ok.any() else np.nan,
                     "extra_share_of_teu_p50": float((extra / teu[ok]).median()) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def s8_rows(fy: pd.DataFrame, key, label, R) -> list[dict]:
    rows = []
    tag1, tag2 = (scen_tag("methanol", "operating", m, 0.0) for m in (1.0, 2.0))
    for subset, q in (("all", fy), ("whole_year", fy[fy["whole_year"]])):
        for grp, p in _groups(q):
            if grp not in ("all", "container"):
                continue
            W = p["w_total"].sum()
            k = p[p["operating_methanol_nm"].notna()]
            Wk = k["w_total"].sum()
            rows.append({"set_key": key, "label": label, "subset": subset, "vessel_group": grp,
                         "n_full": len(p),
                         "touch_w": float(p.loc[p["touches"].astype(bool), "w_total"].sum() / W)
                         if W > 0 else np.nan,
                         "feasible_w": float(p.loc[p["req_lb"] <= R, "w_total"].sum() / W)
                         if W > 0 else np.nan,
                         "covered_lb_w": float(p[f"lb_w_{int(R)}"].sum() / W) if W > 0 else np.nan,
                         "green_p_ub_x1_w": float(k[f"pub_{tag1}"].sum() / Wk) if Wk > 0 else np.nan,
                         "green_p_ub_x2_w": float(k[f"pub_{tag2}"].sum() / Wk) if Wk > 0 else np.nan})
    return rows


def s8_readings(span_rows: pd.DataFrame, strand_all: pd.DataFrame, strand_whole: pd.DataFrame,
                eval_=EVAL, R=RANGE) -> dict:
    f = {}
    c = span_rows[span_rows["vessel_group"] == "container"]
    diffs = []
    for key, cols in (("external:yap4|-1|0|all", ("touch_w", "feasible_w", "green_p_ub_x1_w",
                                                   "green_p_ub_x2_w")),
                      (selected_key("greedy", eval_, R, "all"), ("touch_w", "feasible_w"))):
        a = c[(c["set_key"] == key) & (c["subset"] == "all")]
        w = c[(c["set_key"] == key) & (c["subset"] == "whole_year")]
        if len(a) and len(w):
            for col in cols:
                d = 100 * float(w[col].iloc[0] - a[col].iloc[0])
                diffs.append(abs(d))
                short = "yap4" if "yap4" in key else "greedy"
                f[f"s8_{short}_{col[:-2]}_change_pp"] = d
    if diffs:
        f["s8_max_change_pp"] = max(diffs)
    a = c[(c["set_key"] == "external:yap4|-1|0|all")]
    if len(a):
        n_all = a.loc[a["subset"] == "all", "n_full"]
        n_w = a.loc[a["subset"] == "whole_year", "n_full"]
        if len(n_all) and len(n_w) and int(n_all.iloc[0]):
            f["s8_container_dropped_pct"] = 100 * (1 - int(n_w.iloc[0]) / int(n_all.iloc[0]))

    def overall(st):
        o = st[(st["vessel_group"] == "all") & (st["outcome"] == "A") & (st["metric"] == "overall")] \
            if len(st) else st
        return (float(o["estimate"].iloc[0]), int(o["n_population"].iloc[0])) if len(o) else (np.nan, 0)
    oa, _ = overall(strand_all)
    ow, nw = overall(strand_whole)
    if np.isfinite(oa) and np.isfinite(ow):
        f["s8_stranding_whole_year_pct"] = 100 * ow
        f["s8_stranding_population"] = nw
        f["s8_stranding_change_pp"] = 100 * (ow - oa)
    cells = f.get("s8_max_change_pp", np.nan)
    strand = abs(f.get("s8_stranding_change_pp", np.nan))
    if np.isfinite(cells):
        f["s8_reading"] = ("robust to whole-year observation"
                           if cells <= S8_ROBUST_PP and not strand > S8_ROBUST_PP
                           else "report both in the main text")
    return {k: v for k, v in f.items() if not (isinstance(v, float) and v != v)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--sets", default="port_sets.csv,port_sets_bunker.csv,port_sets_dualfuel.csv",
                    help="port-set files to draw the key sets from (missing ones skipped)")
    ap.add_argument("--train", type=int, default=TRAIN)
    ap.add_argument("--eval", dest="eval_", type=int, default=EVAL)
    ap.add_argument("--range", dest="R", type=int, default=RANGE)
    ap.add_argument("--n", type=int, default=N)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    results = (a.results or out_dir / "results").expanduser().resolve()
    require(out_dir, "stops.csv.gz", "nodes.csv", "vessels.csv", "port_sets.csv",
            "coverage.csv", "redeployment.csv.gz", stage="evaluate.py")
    results.mkdir(parents=True, exist_ok=True)

    stops, nodes, labels = load_inputs(out_dir, extra=("sea_hours",))
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    group_of = vessels.set_index("imo")["group"]
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, a.calibration)
    stops["leg_fuel_t"] = stops["leg_co2_t"] / emissions.CO2_PER_FUEL_T

    e = operating_endurance(vessels, fuel_intensity(stops))
    endur = endurance_operating_summary(e, a.eval_)
    endur_vy, endur_v = endurance_lookup(e, vessels)
    print("S1 endurance at own fuel intensity vs service speed:")
    print(endur[["vessel_group", "n_vessel_years", "ratio_operating_to_service_p50",
                 "operating_methanol_p50", "service_methanol_p50"]].round(2).to_string(index=False))

    prep = chains.prepare(stops, labels, weight_col="leg_co2_t")
    tanks = tank_tonnes(vessels)
    scopes = scope_weights(prep, stops, nodes)
    span = vessel_span(prep)
    del stops
    frames = [pd.read_csv(out_dir / f, dtype={"node_id": str})
              for f in a.sets.split(",") if f and (out_dir / f).exists()]
    port_sets = pd.concat(frames, ignore_index=True).drop_duplicates(["set_key", "rank", "node_id"])
    built = build_sets(port_sets, sorted({a.n, BUNKER_N, *TRADEOFF_NS}), labels)
    red = pd.read_csv(out_dir / "redeployment.csv.gz", dtype={"imo": str})

    def find(key, n):
        return next((x for x in built if x["set_key"] == key and (n < 0 or x["n_ports"] == n)), None)

    own, seg, tank, green, spans, strand, strand_w, space = [], [], [], [], [], None, None, None
    resel = []
    train_key = selected_key("greedy", a.train, a.R, "all")
    for key, n, label in key_sets(a.train, a.eval_, a.R, a.n):
        s = find(key, n)
        if s is None:
            print(f"  {key} (N={n}) not in {a.sets}, skipped")
            continue
        strand_here = key == train_key and n == a.n
        t = set_tables(prep, chains.refuel_mask(prep, s["nodes"]), endur_vy, endur_v,
                       scopes=scopes, strand_R=a.R if strand_here else None, tanks=tanks)
        fy = full_rows(t, a.eval_, group_of)
        fy["whole_year"] = whole_year(fy, span)
        own += [dict(r, n_ports=s["n_ports"]) for r in s1_rows(fy, key, label, a.eval_)]
        seg += s2_rows(fy, key, label, a.eval_)
        tk, gr = s34_rows(fy, key, label, a.eval_)
        tank += tk
        green += gr
        spans += s8_rows(fy, key, label, a.R)
        resel += s11_rows(fy, key, label, s["n_ports"], a.R)
        if key == BASELINE_KEY:
            space = tank_space(fy, vessels)
        if strand_here:
            strand = stranding(t, red, group_of, a.train, a.eval_, a.R)
            strand_w = stranding(t[whole_year(t, span)], red, group_of, a.train, a.eval_, a.R)
        print(f"  {label}: {len(fy):,} fully observed ship-years in {a.eval_} "
              f"({int(fy['whole_year'].sum()):,} seen all year)")
        del t

    extra = [(DUALFUEL_KEY.format(year=a.eval_, R=a.R), 10, "10 chosen for dual-fuel use"),
             (DUALFUEL_KEY.format(year=a.eval_, R=a.R), a.n, f"{a.n} chosen for dual-fuel use"),
             (DUALFUEL_KEY.format(year=a.train, R=a.R), a.n,
              f"{a.n} chosen for dual-fuel use on {a.train}"),
             (selected_key("volume", a.train, 0, "all"), a.n, f"top {a.n} by calls on {a.train}")]
    for key, n, label in extra:
        s = find(key, n)
        if s is None:
            print(f"  S11: {key} (N={n}) not in {a.sets}, skipped")
            continue
        t = set_tables(prep, chains.refuel_mask(prep, s["nodes"]), endur_vy, endur_v,
                       scopes=scopes, tanks=tanks)
        fy = full_rows(t, a.eval_, group_of)
        tk, gr = s34_rows(fy, key, label, a.eval_)
        green += gr
        resel += s11_rows(fy, key, label, n, a.R)
        print(f"  S11 {label}: {len(fy):,} fully observed ship-years in {a.eval_}")
        del t
    overlap = s11_overlap(built, sorted({a.train, a.eval_}), a.R, ns=(10, a.n))

    lat, lon = dt.node_coords(nodes, labels)
    E_stop = stop_endurance(prep, endur_vy, endur_v)
    detour_sets = [("external:yap4|-1|0|all", -1, "Yap hubs (4)"),
                   ("external:yap8|-1|0|all", -1, "Yap hubs (8)"),
                   ("external:bunker10|-1|0|all", -1, "top-10 bunker ports"),
                   (selected_key("greedy", a.eval_, a.R, "all"), a.n, f"{a.n} chosen for continuity")]
    detour_nodes = [(k, lab, find(k, n)["nodes"]) for k, n, lab in detour_sets if find(k, n)]
    dated = results / "methanol_dated_sets.csv"
    if dated.exists():
        ds = pd.read_csv(dated, keep_default_na=False)
        r = ds[(ds["variant"] == "all") & (ds["bound"] == "lower") & (ds["year"] == 2025)]
        if len(r) and r["nodes"].iloc[0]:
            detour_nodes.append(("methanol_dated|lower|2025", "methanol network 2025 (lower)",
                                 r["nodes"].iloc[0].split(";")))
    det_rows = []
    for key, label, node_ids in detour_nodes:
        refuel = chains.refuel_mask(prep, node_ids)
        codes = pd.Index(labels).get_indexer(pd.Index(list(node_ids)))
        det = dt.leg_detour(prep, lat, lon, codes[codes >= 0])
        det_rows += s10_rows(prep, refuel, det, E_stop, a.R, endur_vy, endur_v, group_of,
                             a.eval_, {"set_key": key, "label": label}, tanks=tanks)
        print(f"  S10 {label}: detours computed")
    det_rows = pd.DataFrame(det_rows)

    trade = []
    for fam in TRADEOFF_FAMILIES:
        for ty in sorted({a.train, a.eval_}):
            key = selected_key("greedy", ty, a.R, fam)
            for n in TRADEOFF_NS:
                s = find(key, n)
                if s is None:
                    print(f"  S7: {key} (N={n}) not in {a.sets}, skipped")
                    continue
                t = set_tables(prep, chains.refuel_mask(prep, s["nodes"]), endur_vy, endur_v,
                               ranges=(), scenarios=TRADEOFF_SCEN, scopes=scopes)
                trade += s7_rows(full_rows(t, a.eval_, group_of),
                                 {"set_key": key, "family": fam, "train_year": ty, "n_ports": n})
                del t
    own, seg, tank, green, spans, trade, resel = map(pd.DataFrame,
                                                     (own, seg, tank, green, spans, trade, resel))
    strand = strand if strand is not None else pd.DataFrame()
    strand_w = strand_w if strand_w is not None else pd.DataFrame()

    cov = pd.read_csv(out_dir / "coverage.csv")
    facts = readings(own, seg, tank, green, strand, cov, a.eval_, a.R)
    facts.update(s8_readings(spans, strand, strand_w, a.eval_, a.R))
    facts.update(s10_readings(det_rows))
    facts.update(s11_readings(resel, overlap, a.eval_, a.R, a.n))
    endur.to_csv(results / "endurance_operating.csv", index=False)
    own.to_csv(results / "own_endurance_operating.csv", index=False)
    seg.to_csv(results / "segment_coverage.csv", index=False)
    tank.to_csv(results / "tank_sensitivity.csv", index=False)
    green.to_csv(results / "green_share.csv", index=False)
    strand.to_csv(results / "stranding.csv", index=False)
    strand_w.to_csv(results / "stranding_whole_year.csv", index=False)
    spans.to_csv(results / "observation_span.csv", index=False)
    trade.to_csv(results / "tradeoff.csv", index=False)
    det_rows.to_csv(results / "detour.csv", index=False)
    resel.to_csv(results / "reselection.csv", index=False)
    overlap.to_csv(results / "reselection_overlap.csv", index=False)
    if space is not None:
        space.to_csv(results / "tank_space.csv", index=False)
    emit(results, "supplement", {"train_year": a.train, "eval_year": a.eval_,
                                 "range_nm": a.R, "n_ports": a.n, **facts})
    print("\nreadings:")
    for k, v in facts.items():
        print(f"  {k:<48} {v}")


if __name__ == "__main__":
    main()
