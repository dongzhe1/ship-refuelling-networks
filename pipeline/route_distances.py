from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_JOBS, LEG_ORIGIN, haversine_nm_vec, require
from facts import emit

CHUNK = 2000
MIN_ROUTE_NM = 20.0
SHORTER_THAN_GC = 0.98


def _route_chunk(args):
    import searoute as sr
    out = []
    for key, alat, alon, blat, blon in args:
        try:
            r = sr.searoute((alon, alat), (blon, blat), units="naut")
            out.append((key, float(r.properties["length"]), "ok"))
        except Exception as exc:
            out.append((key, float("nan"), type(exc).__name__))
    return out


def anchorage_positions(stops: pd.DataFrame) -> pd.DataFrame:
    a = stops[["anchorage_id", "lat", "lon"]]
    b = stops[["end_anchorage_id", "lat", "lon"]].rename(
        columns={"end_anchorage_id": "anchorage_id"})
    return (pd.concat([a, b], ignore_index=True).dropna()
              .groupby("anchorage_id")[["lat", "lon"]].median())


def distinct_pairs(stops: pd.DataFrame) -> pd.DataFrame:
    legs = stops[stops["leg_status"] != LEG_ORIGIN]
    return (legs[["leg_from", "anchorage_id"]].dropna()
            .rename(columns={"leg_from": "dep_port", "anchorage_id": "arr_port"})
            .drop_duplicates().reset_index(drop=True))


def read_cache(paths) -> pd.DataFrame:
    frames = []
    for p in paths or []:
        p = Path(p)
        if not p.exists():
            print(f"  cache {p} not found, skipped")
            continue
        c = pd.read_csv(p, dtype={"dep_port": str, "arr_port": str, "status": str})
        c = c[(c["status"] == "ok") & c["routed_nm"].notna() & (c["routed_nm"] > 0)]
        frames.append(c[["dep_port", "arr_port", "routed_nm"]])
        print(f"  cache {p.name}: {len(c):,} usable routed pairs")
    if not frames:
        return pd.DataFrame(columns=["dep_port", "arr_port", "routed_nm"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["dep_port", "arr_port"])


def route_pairs(pairs: pd.DataFrame, jobs: int, router=_route_chunk) -> pd.DataFrame:
    items = [(i, r.dep_lat, r.dep_lon, r.arr_lat, r.arr_lon)
             for i, r in enumerate(pairs.itertuples())]
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    print(f"routing {len(items):,} pairs in {len(chunks):,} chunks on {jobs} process(es)")
    res, st = {}, {}
    if jobs > 1 and len(chunks) > 1:
        with mp.Pool(jobs) as pool:
            for n, out in enumerate(pool.imap_unordered(router, chunks), 1):
                for key, nm, s in out:
                    res[key], st[key] = nm, s
                if n % max(1, len(chunks) // 20) == 0:
                    print(f"  {n}/{len(chunks)} chunks ({100*n/len(chunks):.0f}%)", flush=True)
    else:
        for ch in chunks:
            for key, nm, s in router(ch):
                res[key], st[key] = nm, s
    out = pairs.copy()
    out["routed_nm"] = [res.get(i, np.nan) for i in range(len(out))]
    out["status"] = [st.get(i, "missing") for i in range(len(out))]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--stops", default="stops_gc.csv.gz")
    ap.add_argument("--cache", nargs="*", default=[],
                    help="earlier route_distances.csv files to reuse (read only)")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()

    require(out_dir, a.stops, stage="build_stops.py")
    stops = pd.read_csv(out_dir / a.stops, dtype=str, low_memory=False,
                        usecols=["anchorage_id", "end_anchorage_id", "lat", "lon",
                                 "leg_from", "leg_status"])
    for c in ("lat", "lon"):
        stops[c] = pd.to_numeric(stops[c], errors="coerce")
    pos = anchorage_positions(stops)
    pairs = distinct_pairs(stops)
    del stops
    print(f"{len(pos):,} anchorages located; {len(pairs):,} distinct anchorage pairs")

    pairs = pairs.join(pos.rename(columns={"lat": "dep_lat", "lon": "dep_lon"}), on="dep_port")
    pairs = pairs.join(pos.rename(columns={"lat": "arr_lat", "lon": "arr_lon"}), on="arr_port")
    have = pairs[["dep_lat", "dep_lon", "arr_lat", "arr_lon"]].notna().all(axis=1)
    pairs = pairs[have].reset_index(drop=True)
    pairs["gc_nm"] = haversine_nm_vec(pairs["dep_lat"], pairs["dep_lon"],
                                      pairs["arr_lat"], pairs["arr_lon"])

    cache = read_cache(a.cache)
    pairs = pairs.merge(cache, on=["dep_port", "arr_port"], how="left")
    cached = pairs["routed_nm"].notna()
    short = ~cached & ((pairs["dep_port"] == pairs["arr_port"])
                       | (pairs["gc_nm"] < MIN_ROUTE_NM))
    pairs["status"] = np.where(cached, "ok", np.where(short, "short_kept_gc", ""))
    pairs["source"] = np.where(cached, "cache", np.where(short, "great_circle", "routed"))
    todo = pairs.loc[~cached & ~short].reset_index(drop=True)
    print(f"  {int(cached.sum()):,} pairs from cache, {int(short.sum()):,} short pairs "
          f"keep the great circle, {len(todo):,} to route")

    if len(todo):
        try:
            import searoute
        except ImportError:
            sys.exit("searoute is not installed.\n  pip install searoute  "
                     "(on the LOGIN node; compute nodes have no internet)")
        done = route_pairs(todo.drop(columns=["routed_nm", "status"]), max(1, a.jobs))
        pairs = pd.concat([pairs.loc[cached | short], done], ignore_index=True)

    ok = pairs["routed_nm"].notna() & (pairs["routed_nm"] > 0)
    short = pairs["status"] == "short_kept_gc"
    pairs["ratio"] = pairs["routed_nm"] / pairs["gc_nm"].where(pairs["gc_nm"] > 0)
    snapped = ok & (pairs["ratio"] < SHORTER_THAN_GC) & (pairs["gc_nm"] > 1)
    pairs.loc[snapped, "status"] = "shorter_than_gc"
    ok &= ~snapped
    print(f"\nrouted {int(ok.sum()):,} of {len(pairs):,} ({100*ok.mean():.1f}%); "
          f"{int(short.sum()):,} short pairs kept the great circle")
    bad = ~ok & ~short
    if bad.any():
        print("  failures: " + ", ".join(
            f"{k}={v:,}" for k, v in pairs.loc[bad, "status"].value_counts().items()))
    r = pairs.loc[ok, "ratio"].dropna()
    if len(r):
        print("routed / great-circle: " + "  ".join(
            f"p{int(q*100):02d}={r.quantile(q):.2f}" for q in (.05, .25, .5, .75, .95)))

    out = out_dir / "route_distances.csv"
    pairs[["dep_port", "arr_port", "dep_lat", "dep_lon", "arr_lat", "arr_lon",
           "gc_nm", "routed_nm", "ratio", "status", "source"]].to_csv(out, index=False)
    emit(out_dir, "route_distances", {
        "pairs": int(len(pairs)), "pairs_ok": int(ok.sum()),
        "pairs_from_cache": int((pairs["source"] == "cache").sum()),
        "median_ratio": float(r.median()) if len(r) else 0.0,
    })
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
