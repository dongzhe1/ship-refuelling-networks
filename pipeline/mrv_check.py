from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import emissions
from common import LEG_OK, parse_years, require
from select_ports import load_inputs
from supplement import SCOPES

EEA = SCOPES["eea"]
LARGE_CTR = ["Containership-ULCS", "Containership-Mega Container", "Containership-New Panamax",
             "Containership-Sub New Panamax", "Containership-Post-Panamax"]


def load_mrv_file(path: Path) -> pd.DataFrame:
    raw = (pd.read_excel(path, dtype=str) if path.suffix.lower() == ".xlsx"
           else pd.read_csv(path, dtype=str, sep=None, engine="python", on_bad_lines="skip"))
    cols = {c: re.sub(r"[^a-z0-9]", "", str(c).lower()) for c in raw.columns}

    def pick(*needles, avoid=()):
        for orig, flat in cols.items():
            if any(n in flat for n in needles) and not any(a in flat for a in avoid):
                return orig
        return None
    c_imo = pick("imonumber", "imo")
    c_co2 = pick("totalco2emissions", "totalco2", avoid=("perdistance", "pertransport",
                                                         "onladen", "atberth"))
    c_year = pick("reportingperiod", "year")
    if not (c_imo and c_co2 and c_year):
        raise ValueError(f"no IMO / total CO2 / year column in {list(raw.columns)[:15]}")
    out = pd.DataFrame({
        "imo": raw[c_imo].astype(str).str.extract(r"(\d{7})")[0],
        "reported_co2_t": pd.to_numeric(raw[c_co2].astype(str).str.replace(
            r"[^\d.\-eE]", "", regex=True), errors="coerce"),
        "year": pd.to_numeric(raw[c_year], errors="coerce")})
    return out.dropna()


def load_mrv(path: Path) -> pd.DataFrame:
    files = sorted(list(path.glob("*.csv")) + list(path.glob("*.xlsx"))) if path.is_dir() else [path]
    parts = []
    for f in files:
        try:
            d = load_mrv_file(f)
        except ValueError as exc:
            print(f"  skipped {f.name}: {exc}")
            continue
        print(f"  {f.name}: {len(d):,} rows, years {sorted(d['year'].astype(int).unique())}")
        parts.append(d)
    if not parts:
        sys.exit(f"no readable MRV file under {path}")
    m = pd.concat(parts, ignore_index=True)
    m["year"] = m["year"].astype(int)
    return m.drop_duplicates(["imo", "year"], keep="first")


def modelled(out: Path, calibration: Path | None) -> pd.DataFrame:
    require(out, "stops.csv.gz", "nodes.csv", "vessels.csv", stage="the run")
    stops, nodes, _ = load_inputs(out, extra=("sea_hours",))
    vessels = pd.read_csv(out / "vessels.csv", dtype={"imo": str})
    stops["leg_co2_t"] = emissions.attach_leg_co2(stops, vessels, calibration)
    iso = stops["anchorage_id"].map(nodes.drop_duplicates("anchorage_id")
                                    .set_index("anchorage_id")["iso3"]).fillna("")
    prev_iso = iso.shift(1)
    same = stops["imo"].eq(stops["imo"].shift(1))
    in_scope = same & (iso.isin(EEA) | prev_iso.isin(EEA))
    dep = stops["visit_start"] - pd.to_timedelta(stops["sea_hours"], unit="h")
    keep = in_scope & stops["leg_ok"] & stops["leg_co2_t"].notna()
    legs = pd.DataFrame({"imo": stops.loc[keep, "imo"], "year": dep[keep].dt.year,
                         "co2": stops.loc[keep, "leg_co2_t"]})
    est = legs.groupby(["imo", "year"], as_index=False)["co2"].sum()
    v = vessels.drop_duplicates("imo").set_index("imo")
    est["shiptype_group"] = est["imo"].map(v["shiptype_group"]).fillna("unknown")
    est["group"] = est["imo"].map(v["group"]).fillna("unknown")
    return est


def summary(m: pd.DataFrame, run: str) -> list[dict]:
    rows = []
    for kind in ("shiptype_group", "group"):
        for (y, g), p in m.groupby(["year", kind]):
            r = p["ratio"]
            rows.append({"run": run, "year": int(y), "kind": kind, "group": g, "n": len(p),
                         "median_ratio": float(r.median()), "q25": float(r.quantile(.25)),
                         "q75": float(r.quantile(.75))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", type=Path, nargs="+")
    ap.add_argument("--mrv", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, default=None)
    ap.add_argument("--years", default="2022-2025")
    a = ap.parse_args(argv)
    years = set(parse_years(a.years))
    dest = a.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"MRV: {a.mrv}")
    mrv = load_mrv(a.mrv.expanduser().resolve())
    mrv = mrv[mrv["year"].isin(years) & (mrv["reported_co2_t"] > 0)]
    rows, facts = [], {}
    for out in a.out_dirs:
        out = out.expanduser().resolve()
        run = out.name
        print(f"--- {run}")
        est = modelled(out, a.calibration)
        m = est.merge(mrv, on=["imo", "year"], how="inner")
        m = m[m["co2"] > 0].copy()
        m["ratio"] = m["co2"] / m["reported_co2_t"]
        rows += summary(m, run)
        large = m[m["shiptype_group"].isin(LARGE_CTR)]
        for y in sorted(m["year"].unique()):
            facts[f"{run}|large_container|{y}|median_ratio"] = float(
                large.loc[large["year"] == y, "ratio"].median())
            facts[f"{run}|all|{y}|median_ratio"] = float(m.loc[m["year"] == y, "ratio"].median())
            facts[f"{run}|all|{y}|n"] = int((m["year"] == y).sum())
        print(f"  matched ship-years: {len(m):,}")
    t = pd.DataFrame(rows)
    t.to_csv(dest / "mrv_check.csv", index=False)
    (dest / "facts_mrv_check.json").write_text(json.dumps(facts, indent=1) + "\n")
    show = t[(t["kind"] == "shiptype_group") & t["group"].isin(LARGE_CTR)]
    with pd.option_context("display.width", 200):
        print(show.pivot_table(index=["group", "year"], columns="run",
                               values="median_ratio").round(2).to_string())
        g = t[t["kind"] == "group"]
        print(g.pivot_table(index=["group", "year"], columns="run",
                            values="median_ratio").round(2).to_string())
    print(f"-> {dest}/mrv_check.csv")


if __name__ == "__main__":
    main()
