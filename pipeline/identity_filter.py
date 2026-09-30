from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def filter_visits(v: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = v.copy()
    v["_t"] = pd.to_datetime(v["start"], utc=True, errors="coerce")
    span = (v.groupby(["imo", "vessel_id"])
            .agg(first=("_t", "min"), last=("_t", "max"), n=("_t", "size"),
                 name=("vessel_name", "first")).reset_index())
    op = span.sort_values(["imo", "last", "n"]).groupby("imo").tail(1).set_index("imo")
    v["_op"] = v["imo"].map(op["vessel_id"])
    v["_op_first"] = v["imo"].map(op["first"])
    keep = (v["vessel_id"] == v["_op"]) | (v["_t"] < v["_op_first"])
    report = span.assign(operational=span["vessel_id"] == span["imo"].map(op["vessel_id"]))
    kept = v[keep].groupby(["imo", "vessel_id"]).size()
    report["kept"] = [int(kept.get((i, d), 0)) for i, d in zip(report["imo"], report["vessel_id"])]
    report["dropped"] = report["n"] - report["kept"]
    return v.loc[keep].drop(columns=["_t", "_op", "_op_first"]), report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pull_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    pull, out = a.pull_dir.expanduser().resolve(), a.out_dir.expanduser().resolve()
    if out == pull:
        sys.exit("out_dir must differ from the pull directory (read only)")
    shards = sorted(pull.glob("port_visits_*.csv.gz"))
    if not shards:
        sys.exit(f"no port_visits_*.csv.gz in {pull}")
    v = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in shards],
                  ignore_index=True)
    v = v[v["imo"] != "imo"]
    kept, report = filter_visits(v)
    out.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out / "port_visits_0000.csv.gz", index=False, compression="gzip")
    report.to_csv(out / "identity_filter.csv", index=False)
    multi = report.groupby("imo").size()
    print(f"{len(v):,} visits, {v['imo'].nunique()} ships; {int((multi > 1).sum())} ships with "
          f"more than one identity; {int(report['dropped'].sum())} visits dropped")
    show = report[report["imo"].isin(multi[multi > 1].index)]
    if len(show):
        print(show[["imo", "vessel_id", "name", "first", "last", "operational", "kept",
                    "dropped"]].to_string(index=False))


if __name__ == "__main__":
    main()
