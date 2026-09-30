from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import normalise_imo, require
from facts import emit

SEAWEB_COLS = ["lrnoimo_ship_no", "shiptype_level5", "shiptype_group", "ship_status",
               "deadweight",
               "gross_tonnage", "teu", "year_of_build", "speedservice",
               "total_kilowattsof_main_engines", "bunkers_descriptive_narrative",
               "last_update_date"]

GROUPS = ["container", "bulk", "tanker", "gas", "general", "roro", "passenger", "other"]

_CAP = re.compile(r"([\d][\d,]*(?:\.\d+)?)\s*cu m")
_CONS = re.compile(r"consumption:\s*([\d]+(?:\.\d+)?)\s*tonnes per day")


def ship_group(level5) -> str:
    if not isinstance(level5, str) or not level5.strip():
        return "other"
    s = level5.lower()
    if "container" in s:
        return "container"
    if "lng" in s or "lpg" in s or "gas" in s:
        return "gas"
    if "tanker" in s:
        return "tanker"
    if "bulk" in s or "ore carrier" in s or "cement carrier" in s or "wood chips" in s:
        return "bulk"
    if "passenger" in s or "cruise" in s:
        return "passenger"
    if "vehicle" in s or "ro-ro" in s:
        return "roro"
    if ("general cargo" in s or "refrigerated" in s or "multi-purpose" in s
            or "heavy load" in s or "open hatch" in s):
        return "general"
    return "other"


def parse_bunkers(text):
    if not isinstance(text, str):
        return np.nan, np.nan
    caps = [float(x.replace(",", "")) for x in _CAP.findall(text)]
    cons = _CONS.findall(text)
    return (sum(caps) if caps else np.nan,
            float(cons[0]) if cons else np.nan)


def build(seaweb: pd.DataFrame, imos=None) -> pd.DataFrame:
    sw = seaweb.copy()
    sw["imo"] = normalise_imo(sw["lrnoimo_ship_no"])
    sw = sw.dropna(subset=["imo"]).drop_duplicates("imo")
    if imos is not None:
        sw = sw[sw["imo"].isin(set(imos))]
    parsed = sw["bunkers_descriptive_narrative"].map(parse_bunkers) \
        if "bunkers_descriptive_narrative" in sw else pd.Series([(np.nan, np.nan)] * len(sw))
    out = pd.DataFrame({
        "imo": sw["imo"].to_numpy(),
        "group": sw["shiptype_level5"].map(ship_group).to_numpy(),
        "shiptype_level5": sw["shiptype_level5"].to_numpy(),
        "shiptype_group": sw["shiptype_group"].to_numpy() if "shiptype_group" in sw
        else np.full(len(sw), None, dtype=object),
        "ship_status": sw.get("ship_status", pd.Series(index=sw.index, dtype=str)).to_numpy(),
        "fuel_capacity_m3": [p[0] for p in parsed],
        "consumption_tpd": [p[1] for p in parsed],
    })
    for src, dst in [("deadweight", "dwt"), ("gross_tonnage", "gt"), ("teu", "teu"),
                     ("year_of_build", "built"), ("speedservice", "service_kn"),
                     ("total_kilowattsof_main_engines", "main_kw")]:
        out[dst] = pd.to_numeric(sw[src], errors="coerce").to_numpy() if src in sw \
            else np.nan
    for c in ("fuel_capacity_m3", "consumption_tpd", "service_kn", "main_kw"):
        out.loc[out[c] <= 0, c] = np.nan
    return out.sort_values("imo").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--seaweb", type=Path, required=True)
    ap.add_argument("--stops", default="stops.csv.gz")
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()

    require(out_dir, a.stops)
    if not a.seaweb.exists():
        sys.exit(f"Sea-web file not found: {a.seaweb} (set GN_SEAWEB)")
    head = pd.read_csv(a.seaweb, nrows=0).columns
    cols = [c for c in SEAWEB_COLS if c in head]
    if "lrnoimo_ship_no" not in cols:
        sys.exit(f"{a.seaweb}: no lrnoimo_ship_no column")
    sw = pd.read_csv(a.seaweb, usecols=cols, dtype=str, low_memory=False)
    if "last_update_date" in sw:
        print(f"Sea-web last_update_date max: {sw['last_update_date'].max()}")
    imos = pd.read_csv(out_dir / a.stops, usecols=["imo"], dtype=str)["imo"].unique()
    v = build(sw, imos)

    matched = len(v) / max(len(imos), 1)
    print(f"{len(v):,} of {len(imos):,} stop-table vessels matched ({100*matched:.1f}%)")
    print(v["group"].value_counts().to_string())
    cap = v["fuel_capacity_m3"].notna()
    print(f"fuel capacity recorded for {100*cap.mean():.1f}%, "
          f"consumption for {100*v['consumption_tpd'].notna().mean():.1f}%")

    out = out_dir / "vessels.csv"
    v.to_csv(out, index=False)
    facts = {"vessels_matched": int(len(v)), "vessels_in_stops": int(len(imos)),
             "fuel_capacity_pct": 100.0 * float(cap.mean()) if len(v) else 0.0}
    for g in GROUPS:
        facts[f"n_{g}"] = int((v["group"] == g).sum())
    emit(out_dir, "vessels", facts)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
