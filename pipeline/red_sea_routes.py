from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFAULT_JOBS, require
from facts import emit

CHUNK = 2000
MIN_NM = 1000.0
RED_SEA = {"suez", "babalmandab"}
CLOSED = ["northwest", "suez", "babalmandab"]


def _route_chunk(args):
    import warnings
    import searoute as sr
    out = []
    for key, alat, alon, blat, blon in args:
        try:
            r = sr.searoute((alon, alat), (blon, blat), units="naut", return_passages=True)
            passages = set(r.properties.get("traversed_passages") or [])
            if not RED_SEA <= passages:
                out.append((key, False, float(r.properties["length"]), np.nan, "ok"))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = sr.searoute((alon, alat), (blon, blat), units="naut", restrictions=CLOSED)
            cape = float(c.properties["length"])
            out.append((key, True, float(r.properties["length"]),
                        cape if cape > 0 else np.nan, "ok" if cape > 0 else "no_cape_route"))
        except Exception as exc:
            out.append((key, False, np.nan, np.nan, type(exc).__name__))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--min-nm", type=float, default=MIN_NM)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "route_distances.csv", stage="route_distances.py")
    try:
        import searoute
    except ImportError:
        sys.exit("searoute is not installed.\n  pip install searoute")
    rd = pd.read_csv(out_dir / "route_distances.csv", dtype={"dep_port": str, "arr_port": str})
    pairs = rd[(rd["status"] == "ok") & (rd["routed_nm"] >= a.min_nm)].reset_index(drop=True)
    items = [(i, r.dep_lat, r.dep_lon, r.arr_lat, r.arr_lon) for i, r in enumerate(pairs.itertuples())]
    chunks = [items[i:i + CHUNK] for i in range(0, len(items), CHUNK)]
    print(f"{len(rd):,} pairs; {len(items):,} of at least {a.min_nm:,.0f} nm to check, "
          f"{len(chunks):,} chunks on {a.jobs} process(es)")
    res = {}
    if a.jobs > 1 and len(chunks) > 1:
        with mp.Pool(a.jobs) as pool:
            for n, out in enumerate(pool.imap_unordered(_route_chunk, chunks), 1):
                res.update({k: v for k, *v in out})
                if n % max(1, len(chunks) // 20) == 0:
                    print(f"  {n}/{len(chunks)} chunks", flush=True)
    else:
        for ch in chunks:
            res.update({k: v for k, *v in _route_chunk(ch)})
    cols = ["red_sea", "default_nm", "cape_nm", "status"]
    pairs[cols] = pd.DataFrame([res.get(i, [False, np.nan, np.nan, "missing"])
                                for i in range(len(pairs))], columns=cols)
    red = pairs[pairs["red_sea"].astype(bool)]
    ok = red["cape_nm"].notna()
    print(f"{len(red):,} pairs cross the Red Sea; {int(ok.sum()):,} with a route round the Cape")
    if ok.any():
        r = red.loc[ok, "cape_nm"] / red.loc[ok, "routed_nm"]
        print("cape / routed: " + "  ".join(f"p{int(q*100):02d}={r.quantile(q):.2f}"
                                            for q in (.05, .25, .5, .75, .95)))
    out = out_dir / "red_sea_routes.csv"
    red[["dep_port", "arr_port", "routed_nm", "default_nm", "cape_nm", "status"]].to_csv(out, index=False)
    emit(out_dir, "red_sea_routes", {"pairs_checked": int(len(pairs)), "pairs_red_sea": int(len(red)),
                                     "pairs_cape": int(ok.sum())})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
