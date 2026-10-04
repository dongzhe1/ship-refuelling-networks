from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LEG_OK, require
from facts import emit

RED = ("routed_red_sea", "routed_cape")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "stops.csv.gz", "vessels.csv", "red_sea_routes.csv", stage="build_stops.py --cape")
    s = pd.read_csv(out_dir / "stops.csv.gz", dtype={"imo": str}, low_memory=False,
                    usecols=["imo", "visit_start", "leg_from", "anchorage_id", "leg_nm",
                             "distance_source", "leg_status"])
    s = s[s["distance_source"].isin(RED) & (s["leg_status"] == LEG_OK)].copy()
    s["year"] = pd.to_datetime(s["visit_start"], utc=True, format="mixed").dt.year
    v = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    s["vessel_group"] = s["imo"].map(v.drop_duplicates("imo").set_index("imo")["group"]).fillna("unknown")
    r = pd.read_csv(out_dir / "red_sea_routes.csv", dtype={"dep_port": str, "arr_port": str})
    s = s.merge(r[["dep_port", "arr_port", "routed_nm", "cape_nm"]].drop_duplicates(["dep_port", "arr_port"]),
                left_on=["leg_from", "anchorage_id"], right_on=["dep_port", "arr_port"], how="left")
    s["cape"] = s["distance_source"] == "routed_cape"
    s["added_nm"] = (s["cape_nm"] - s["routed_nm"]).where(s["cape"], 0.0)
    rows = []
    for (y, g), p in pd.concat([s, s.assign(vessel_group="all")]).groupby(["year", "vessel_group"]):
        rows.append({"year": int(y), "vessel_group": g, "legs_red_sea": len(p),
                     "legs_cape": int(p["cape"].sum()), "cape_share": float(p["cape"].mean()),
                     "ships": int(p["imo"].nunique()), "ships_cape": int(p.loc[p["cape"], "imo"].nunique()),
                     "added_nm": float(p["added_nm"].sum())})
    t = pd.DataFrame(rows)
    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    t.to_csv(res / "cape_routing.csv", index=False)
    show = t[t["vessel_group"].isin(["all", "container", "tanker", "bulk"])]
    print(show.pivot(index="year", columns="vessel_group", values="cape_share").round(3).to_string())
    facts = {f"cape_share_{r.vessel_group}_{r.year}": float(r.cape_share) for r in t.itertuples()}
    facts.update({f"legs_red_sea_{r.vessel_group}_{r.year}": int(r.legs_red_sea) for r in t.itertuples()})
    emit(res, "cape_routing", facts)


if __name__ == "__main__":
    main()
