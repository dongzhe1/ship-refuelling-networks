from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEFAULT_JOBS, LEG_BREAK_GAP, LEG_BREAK_SPEED, LEG_OK,
                    LEG_ORIGIN, MAX_DURATION_HRS, MAX_IMPLIED_KN, MIN_CONFIDENCE,
                    haversine_nm_vec, leg_status, normalise_imo, to_utc)
from facts import emit

HERE = Path(__file__).resolve().parent
TRANSIT_FILE = HERE / "reference" / "transit_anchorage_ids.txt"

WANT = ["event_id", "imo", "start", "end", "confidence", "lat", "lon",
        "start_anchorage_id", "start_anchorage_name", "start_anchorage_flag",
        "start_at_dock", "end_anchorage_id"]
REQUIRED = {"imo", "start", "end", "lat", "lon", "start_anchorage_id",
            "end_anchorage_id"}

DETOUR_MAX_GC_NM = 100.0
DETOUR_MAX_RATIO = 3.0

OUT_COLS = ["imo", "visit_start", "visit_end", "anchorage_id", "end_anchorage_id",
            "iso3", "name", "lat", "lon", "at_dock", "transit", "leg_from",
            "sea_hours", "leg_gc_nm", "leg_nm", "distance_source", "implied_kn",
            "leg_status", "slow"]


def _read_shard(path):
    head = pd.read_csv(path, nrows=0).columns
    cols = [c for c in WANT if c in head]
    missing = REQUIRED - set(cols)
    if missing:
        raise SystemExit(f"{path.name}: missing columns {sorted(missing)}")
    df = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
    return df[df["imo"] != "imo"].copy()


def load_visits(shards, jobs):
    read_jobs = max(1, min(jobs, len(shards)))
    print(f"reading {len(shards)} shard(s) on {read_jobs} process(es)")
    if read_jobs > 1:
        with mp.Pool(read_jobs) as pool:
            frames = pool.map(_read_shard, shards)
    else:
        frames = [_read_shard(s) for s in shards]
    df = pd.concat(frames, ignore_index=True)
    del frames
    counts = {"rows_read": len(df)}

    df["imo"] = normalise_imo(df["imo"])
    df["start"] = to_utc(df["start"])
    df["end"] = to_utc(df["end"])
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[["imo", "start", "end", "lat", "lon"]].isna().any(axis=1)
    counts["dropped_unparseable"] = int(bad.sum())
    df = df[~bad]

    if "confidence" in df.columns:
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        low = conf.lt(MIN_CONFIDENCE).to_numpy()
        counts["dropped_low_confidence"] = int(low.sum())
        df = df[~low]

    dur = (df["end"] - df["start"]).dt.total_seconds() / 3600.0
    long_or_neg = ((dur > MAX_DURATION_HRS) | (dur < 0)).to_numpy()
    counts["dropped_duration"] = int(long_or_neg.sum())
    df = df[~long_or_neg]

    before = len(df)
    df = df.drop_duplicates(subset=["imo", "start", "start_anchorage_id"])
    counts["dropped_duplicate"] = before - len(df)

    df = df.sort_values(["imo", "start"], kind="mergesort").reset_index(drop=True)
    counts["visits"] = len(df)
    return df, counts


def build(df: pd.DataFrame, routes: pd.DataFrame | None = None,
          transit_ids: set[str] | None = None) -> pd.DataFrame:
    imo = df["imo"].to_numpy()
    first = np.r_[True, imo[1:] != imo[:-1]]

    prev_end = df["end"].shift(1)
    prev_anch = df["end_anchorage_id"].shift(1)
    prev_lat = df["lat"].shift(1).to_numpy(float)
    prev_lon = df["lon"].shift(1).to_numpy(float)

    sea_hours = ((df["start"] - prev_end).dt.total_seconds() / 3600.0).to_numpy(float, copy=True)
    sea_hours[first] = np.nan
    gc = haversine_nm_vec(prev_lat, prev_lon, df["lat"].to_numpy(float),
                          df["lon"].to_numpy(float))
    gc[first] = np.nan

    out = pd.DataFrame({
        "imo": df["imo"].to_numpy(),
        "visit_start": df["start"].reset_index(drop=True),
        "visit_end": df["end"].reset_index(drop=True),
        "anchorage_id": df["start_anchorage_id"].to_numpy(),
        "end_anchorage_id": df["end_anchorage_id"].to_numpy(),
        "iso3": df["start_anchorage_flag"].to_numpy() if "start_anchorage_flag" in df else None,
        "name": df["start_anchorage_name"].to_numpy() if "start_anchorage_name" in df else None,
        "lat": df["lat"].to_numpy(float),
        "lon": df["lon"].to_numpy(float),
        "at_dock": df["start_at_dock"].to_numpy() if "start_at_dock" in df else None,
    })
    leg_from = prev_anch.to_numpy(dtype=object, copy=True)
    leg_from[first] = None
    out["leg_from"] = leg_from
    out["sea_hours"] = sea_hours
    out["leg_gc_nm"] = gc
    out["leg_nm"] = gc
    out["distance_source"] = np.where(first, "", "great_circle")

    if routes is not None:
        usable = routes[(routes["status"] == "ok") & routes["routed_nm"].notna()
                        & (routes["routed_nm"] > 0)]
        usable = usable.drop_duplicates(subset=["dep_port", "arr_port"])
        m = out[["leg_from", "anchorage_id"]].merge(
            usable[["dep_port", "arr_port", "routed_nm"]],
            left_on=["leg_from", "anchorage_id"], right_on=["dep_port", "arr_port"],
            how="left")
        routed = m["routed_nm"].to_numpy(float)
        hit = ~np.isnan(routed) & ~first
        with np.errstate(divide="ignore", invalid="ignore"):
            too_fast = hit & (sea_hours > 0) & (routed / sea_hours > MAX_IMPLIED_KN) \
                & (gc / sea_hours <= MAX_IMPLIED_KN)
        detour = hit & ~too_fast & (gc < DETOUR_MAX_GC_NM) & (routed > DETOUR_MAX_RATIO * gc)
        use = hit & ~too_fast & ~detour
        out.loc[use, "leg_nm"] = routed[use]
        out.loc[use, "distance_source"] = "routed"
        out.loc[too_fast, "distance_source"] = "great_circle_route_too_fast"
        out.loc[detour, "distance_source"] = "great_circle_route_detour"

    with np.errstate(divide="ignore", invalid="ignore"):
        out["implied_kn"] = np.where(out["sea_hours"] > 0,
                                     out["leg_nm"] / out["sea_hours"], np.nan)
    status, slow = leg_status(gc, sea_hours)
    out["leg_status"] = status
    out["slow"] = slow
    ids = transit_ids or set()
    out["transit"] = out["anchorage_id"].astype(str).isin(ids).to_numpy()
    return out[OUT_COLS]


def read_transit_ids(path: Path = TRANSIT_FILE) -> set[str]:
    if not path.exists():
        return set()
    return {l.strip() for l in path.read_text().split() if l.strip()}


def summarise(stops: pd.DataFrame, counts: dict) -> dict:
    legs = stops[stops["leg_status"] != LEG_ORIGIN]
    n_legs = max(len(legs), 1)
    vc = legs["leg_status"].value_counts()
    facts = dict(counts)
    facts.update({
        "vessels": int(stops["imo"].nunique()),
        "anchorages": int(stops["anchorage_id"].nunique()),
        "legs": int(len(legs)),
        "legs_ok": int(vc.get(LEG_OK, 0)),
        "legs_break_gap": int(vc.get(LEG_BREAK_GAP, 0)),
        "legs_break_speed": int(vc.get(LEG_BREAK_SPEED, 0)),
        "legs_slow": int(legs["slow"].sum()),
        "legs_routed_pct": 100.0 * float((legs["distance_source"] == "routed").mean())
        if len(legs) else 0.0,
        "legs_route_too_fast": int((legs["distance_source"] == "great_circle_route_too_fast").sum()),
        "legs_route_detour": int((legs["distance_source"] == "great_circle_route_detour").sum()),
        "stops_at_transit_pct": 100.0 * float(stops["transit"].mean()) if len(stops) else 0.0,
        "median_leg_nm": float(legs.loc[legs["leg_status"] == LEG_OK, "leg_nm"].median())
        if (legs["leg_status"] == LEG_OK).any() else 0.0,
    })
    print(f"\nstops: {len(stops):,}  vessels: {facts['vessels']:,}  "
          f"anchorages: {facts['anchorages']:,}")
    print(f"legs:  {len(legs):,}")
    for k in ("legs_ok", "legs_break_gap", "legs_break_speed", "legs_slow"):
        print(f"  {k:<18} {facts[k]:>10,}  ({100*facts[k]/n_legs:.2f}%)")
    print(f"  routed distance on {facts['legs_routed_pct']:.1f}% of legs; "
          f"great circle kept for {facts['legs_route_too_fast']:,} routes impossible "
          f"in the time taken and {facts['legs_route_detour']:,} short-hop detours")
    print(f"  stops at the transit anchorages: {facts['stops_at_transit_pct']:.2f}%"
          f"  (kept here; reference drops them)")
    return facts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--gfw", type=Path, required=True,
                    help="directory holding port_visits_*.csv.gz (read only)")
    ap.add_argument("--routes", type=Path, default=None,
                    help="route_distances.csv from route_distances.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    a = ap.parse_args(argv)

    out_dir = a.out_dir.expanduser().resolve()
    gfw = a.gfw.expanduser().resolve()
    if out_dir == gfw:
        sys.exit("out_dir must not be the GFW directory: the outputs live there")
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(gfw.glob("port_visits_*.csv.gz"))
    if not shards:
        sys.exit(f"no port_visits_*.csv.gz in {gfw}")

    df, counts = load_visits(shards, max(1, a.jobs))
    print(f"  {counts['visits']:,} visits kept of {counts['rows_read']:,} rows read")
    for k, v in counts.items():
        if k.startswith("dropped_"):
            print(f"    {k:<24} {v:>10,}")

    routes = None
    if a.routes is not None:
        routes = pd.read_csv(a.routes, dtype={"dep_port": str, "arr_port": str,
                                              "status": str})
        print(f"routes: {len(routes):,} anchorage pairs from {a.routes.name}")

    stops = build(df, routes, read_transit_ids())
    del df
    facts = summarise(stops, counts)

    name = a.out or ("stops.csv.gz" if routes is not None else "stops_gc.csv.gz")
    out = out_dir / name
    stops.to_csv(out, index=False, compression="gzip")
    emit(out_dir, Path(name).name.split(".")[0], facts)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
