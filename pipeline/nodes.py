from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import haversine_km_vec, require
from facts import emit

HERE = Path(__file__).resolve().parent
OVERRIDES = HERE / "reference" / "node_overrides.csv"

MERGE_KM = 30.0


def anchorage_table(stops: pd.DataFrame) -> pd.DataFrame:
    g = stops.groupby("anchorage_id", sort=True)
    t = pd.DataFrame({
        "visits": g.size(),
        "lat": g["lat"].median(),
        "lon": g["lon"].median(),
    })
    for c in ("iso3", "name"):
        if c in stops.columns:
            t[c] = g[c].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
        else:
            t[c] = ""
    return t.reset_index()


def assign_nodes(anch: pd.DataFrame, merge_km: float = MERGE_KM) -> pd.DataFrame:
    a = anch.copy()
    a["iso3"] = a["iso3"].fillna("").astype(str)
    a["node_id"] = None
    a["how"] = None
    a["km_to_node"] = np.nan
    a = a.sort_values(["visits", "anchorage_id"], ascending=[False, True]).reset_index(drop=True)
    for iso, grp in a.groupby("iso3", sort=True):
        idx = grp.index.to_numpy()
        lat = a.loc[idx, "lat"].to_numpy(float)
        lon = a.loc[idx, "lon"].to_numpy(float)
        free = np.ones(len(idx), bool)
        for k in range(len(idx)):
            if not free[k]:
                continue
            leader = a.at[idx[k], "anchorage_id"]
            if not iso or np.isnan(lat[k]):
                cand = np.zeros(len(idx), bool)
                cand[k] = True
                d = np.zeros(len(idx))
            else:
                d = haversine_km_vec(lat[k], lon[k], lat, lon)
                cand = free & (d <= merge_km)
                cand[k] = True
            members = idx[cand]
            a.loc[members, "node_id"] = leader
            a.loc[members, "how"] = "radius"
            a.loc[members, "km_to_node"] = d[cand]
            a.at[idx[k], "how"] = "leader"
            a.at[idx[k], "km_to_node"] = 0.0
            free &= ~cand
    return a


def apply_overrides(a: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    if ov is None or ov.empty:
        return a
    a = a.set_index("anchorage_id")
    node_of = a["node_id"].to_dict()
    for r in ov.itertuples():
        if r.anchorage_id not in a.index:
            print(f"  override: {r.anchorage_id} not in data, skipped")
            continue
        target = node_of.get(r.target_anchorage_id)
        if target is None:
            print(f"  override: target {r.target_anchorage_id} not in data, skipped")
            continue
        a.at[r.anchorage_id, "node_id"] = target
        a.at[r.anchorage_id, "how"] = "override"
    return a.reset_index()


def read_overrides(path: Path = OVERRIDES) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["anchorage_id", "target_anchorage_id", "reason"])
    return pd.read_csv(path, dtype=str, comment="#").dropna(
        subset=["anchorage_id", "target_anchorage_id"])


def finish(a: pd.DataFrame) -> pd.DataFrame:
    lead = a.set_index("anchorage_id")
    a = a.copy()
    a["node_name"] = a["node_id"].map(lead["name"])
    a["node_iso3"] = a["node_id"].map(lead["iso3"])
    a["node_lat"] = a["node_id"].map(lead["lat"])
    a["node_lon"] = a["node_id"].map(lead["lon"])
    a["node_visits"] = a.groupby("node_id")["visits"].transform("sum")
    cols = ["anchorage_id", "name", "iso3", "lat", "lon", "visits", "node_id",
            "node_name", "node_iso3", "node_lat", "node_lon", "node_visits",
            "how", "km_to_node"]
    return a[cols].sort_values(["node_visits", "node_id", "visits"],
                               ascending=[False, True, False]).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stops", default="stops.csv.gz")
    ap.add_argument("--merge-km", type=float, default=MERGE_KM)
    ap.add_argument("--overrides", type=Path, default=OVERRIDES)
    ap.add_argument("--out", default="nodes.csv")
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()

    require(out_dir, a.stops)
    stops = pd.read_csv(out_dir / a.stops, dtype=str, low_memory=False,
                        usecols=["anchorage_id", "iso3", "name", "lat", "lon"])
    for c in ("lat", "lon"):
        stops[c] = pd.to_numeric(stops[c], errors="coerce")
    anch = anchorage_table(stops)
    del stops
    nodes = finish(apply_overrides(assign_nodes(anch, a.merge_km),
                                   read_overrides(a.overrides)))

    n_nodes = nodes["node_id"].nunique()
    merged = nodes[nodes["how"] != "leader"]
    print(f"{len(nodes):,} anchorages -> {n_nodes:,} nodes at {a.merge_km:g} km "
          f"({len(merged):,} absorbed)")
    top = (nodes.drop_duplicates("node_id").head(15))
    print("largest nodes (members):")
    for r in top.itertuples():
        members = nodes.loc[nodes["node_id"] == r.node_id, "anchorage_id"].tolist()
        more = f" +{len(members) - 4}" if len(members) > 4 else ""
        print(f"  {r.node_id:<32} {r.node_visits:>9,}  {', '.join(members[:4])}{more}")

    out = out_dir / a.out
    nodes.to_csv(out, index=False)
    emit(out_dir, Path(a.out).name.split(".")[0], {"anchorages": int(len(nodes)), "nodes": int(n_nodes),
                            "merge_km": a.merge_km, "absorbed": int(len(merged))})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
