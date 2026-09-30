from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot as pl
from common import require
from evaluate import BASELINE_KEY
from facts import emit

EVAL, TRAIN, RANGE, N = 2024, 2019, 7000, 20
ROBUST_PP = 5.0
COMPARED = [("external:yap4|-1|0|all", -1, "yap4"), ("external:yap8|-1|0|all", -1, "yap8"),
            (f"greedy|{EVAL}|{RANGE}|all", N, "greedy_eval"),
            (f"greedy|{TRAIN}|{RANGE}|all", N, "greedy_train")]


def full(sy: pd.DataFrame) -> pd.DataFrame:
    return sy[(sy["n_breaks"] == 0) & sy["req_lb"].notna()]


def touched(sy: pd.DataFrame) -> pd.Series:
    t = sy["touches"]
    return t if t.dtype == bool else t.astype(str).eq("True")


def cells(sy: pd.DataFrame, R: int) -> dict:
    W = sy["w_total"].sum()
    return {"n_full": int(len(sy)), "w_total": float(W),
            "touch_w": float(sy.loc[touched(sy), "w_total"].sum() / W) if W > 0 else np.nan,
            "feasible_w": float(sy.loc[sy["req_lb"] <= R, "w_total"].sum() / W) if W > 0 else np.nan,
            "median_req_lb": float(sy["req_lb"].median()) if len(sy) else np.nan}


def stranding(sy: pd.DataFrame, train: int, eval_: int, R: int) -> tuple[float, int]:
    f = full(sy)
    a = f[f["year"] == train].set_index("imo")["req_lb"]
    b = f[f["year"] == eval_].set_index("imo")["req_lb"]
    j = pd.concat([a.rename("t"), b.rename("y")], axis=1, join="inner")
    pop = j[j["t"] <= R]
    return (float((pop["y"] > R).mean()) if len(pop) else np.nan), int(len(pop))


def compare(main: pd.DataFrame, rerun: pd.DataFrame, eval_: int, train: int, R: int):
    rows, f = [], {}
    for key, n, short in COMPARED + [(BASELINE_KEY, -1, "all_nodes")]:
        m = full(pl.rows_of(main, key, n))
        r = full(pl.rows_of(rerun, key, n))
        me, re_ = m[m["year"] == eval_], r[r["year"] == eval_]
        both = me.merge(re_[["imo"]], on="imo")[["imo"]]
        for scope, mm, rr in (("own", me, re_),
                              ("both", me.merge(both, on="imo"), re_.merge(both, on="imo"))):
            cm, cr = cells(mm, R), cells(rr, R)
            rows.append({"set": short, "set_key": key, "scope": scope,
                         **{f"main_{k}": v for k, v in cm.items()},
                         **{f"all_ids_{k}": v for k, v in cr.items()}})
    t = pd.DataFrame(rows)
    own = t[(t["scope"] == "own") & (t["set"] != "all_nodes")]
    diffs = []
    for _, r in own.iterrows():
        for col in ("touch_w", "feasible_w"):
            d = 100 * (r[f"all_ids_{col}"] - r[f"main_{col}"])
            f[f"{r['set']}_{col[:-2]}_change_pp"] = d
            diffs.append(abs(d))
    for scope in ("own",):
        a = t[(t["scope"] == scope) & (t["set"] == "all_nodes")].iloc[0]
        f["n_full_main"], f["n_full_all_ids"] = int(a["main_n_full"]), int(a["all_ids_n_full"])
        f["co2_gain_pct"] = 100 * (a["all_ids_w_total"] / a["main_w_total"] - 1) \
            if a["main_w_total"] > 0 else np.nan
    key = f"greedy|{train}|{R}|all"
    sm, nm = stranding(pl.rows_of(main, key, N), train, eval_, R)
    sr, nr = stranding(pl.rows_of(rerun, key, N), train, eval_, R)
    f.update({"strand_main_pct": 100 * sm, "strand_all_ids_pct": 100 * sr,
              "strand_main_n": nm, "strand_all_ids_n": nr,
              "strand_change_pp": 100 * (sr - sm)})
    if diffs:
        f["max_change_pp"] = max(diffs)
        strand_ok = not abs(f["strand_change_pp"]) > ROBUST_PP
        f["reading"] = ("robust to the identity rule" if max(diffs) <= ROBUST_PP and strand_ok
                        else "report and discuss in the main text")
    return t, {k: v for k, v in f.items() if not (isinstance(v, float) and v != v)}


def spans(path: Path, imos: set) -> pd.DataFrame:
    parts = []
    for ch in pd.read_csv(path, usecols=["imo", "visit_start"], dtype={"imo": str},
                          chunksize=2_000_000):
        ch = ch[ch["imo"].isin(imos)]
        if len(ch):
            t = pd.to_datetime(ch["visit_start"], utc=True).dt.tz_localize(None)
            parts.append(pd.DataFrame({"imo": ch["imo"], "t": t}))
    if not parts:
        return pd.DataFrame(columns=["first", "last"])
    d = pd.concat(parts, ignore_index=True)
    return d.groupby("imo")["t"].agg(first="min", last="max")


def whole_year_rows(sy: pd.DataFrame, span: pd.DataFrame) -> pd.DataFrame:
    f = sy["imo"].map(span["first"])
    l = sy["imo"].map(span["last"])
    y = sy["year"].astype(str)
    ok = (f <= pd.to_datetime(y + "-01-31 23:59:59")) & (l >= pd.to_datetime(y + "-12-01"))
    return sy[ok.to_numpy()]


def cross_check(main, rerun, span_main, span_rerun, train, eval_, R) -> dict:
    f = {}
    key = f"greedy|{train}|{R}|all"
    for name, sy, span in (("main", main, span_main), ("all_ids", rerun, span_rerun)):
        rows = pl.rows_of(sy, key, N)
        for scope, part in (("all", rows), ("whole", whole_year_rows(rows, span))):
            share, n = stranding(part, train, eval_, R)
            if np.isfinite(share):
                f[f"x_{name}_{scope}_strand_pct"] = 100 * share
                f[f"x_{name}_{scope}_strand_n"] = n
    return f


def years_table(main: pd.DataFrame, rerun: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sy in (("main", main), ("all_ids", rerun)):
        b = full(pl.rows_of(sy, BASELINE_KEY, -1))
        g = b.groupby("year").agg(n_full=("imo", "size"), ships=("imo", "nunique"),
                                  w_total=("w_total", "sum")).reset_index()
        rows.append(g.assign(version=name))
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rerun_dir", type=Path)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--ships", type=Path, default=None, help="default <rerun_dir>/ships.csv")
    ap.add_argument("--eval", dest="eval_", type=int, default=EVAL)
    ap.add_argument("--train", type=int, default=TRAIN)
    ap.add_argument("--range", dest="R", type=int, default=RANGE)
    a = ap.parse_args(argv)
    rr = a.rerun_dir.expanduser().resolve()
    require(rr, "ship_years.csv.gz", stage="evaluate.py on the rerun directory")
    ships = pl.read_ships(a.ships or rr / "ships.csv")
    imos = set(ships["imo"])
    keys = {k for k, _, _ in COMPARED} | {BASELINE_KEY}
    years = sorted({a.train, a.eval_} | set(range(2018, 2026)))
    rerun = pd.read_csv(rr / "ship_years.csv.gz", dtype={"imo": str})
    rerun = rerun[rerun["imo"].isin(imos) & rerun["set_key"].isin(keys)
                  & (rerun["vessel_group"] == "container")]
    main_sy = pl.read_main_ship_years(a.main.expanduser().resolve(), keys, years,
                                      group="container", extra=("w_total",))
    main_sy = main_sy[main_sy["imo"].isin(imos)]
    t, f = compare(main_sy, rerun, a.eval_, a.train, a.R)
    if (rr / "stops.csv.gz").exists() and (a.main / "stops.csv.gz").exists():
        f.update(cross_check(main_sy, rerun, spans(a.main / "stops.csv.gz", imos),
                             spans(rr / "stops.csv.gz", imos), a.train, a.eval_, a.R))
    yt = years_table(main_sy, rerun)
    res = rr / "results"
    res.mkdir(exist_ok=True)
    t.to_csv(res / "identity_rerun.csv", index=False)
    yt.to_csv(res / "identity_rerun_years.csv", index=False)
    idf = rr / "pull_clean" / "identity_filter.csv"
    if idf.exists():
        d = pd.read_csv(idf, dtype={"imo": str})
        multi = d.groupby("imo").size()
        f["ships_multi_identity"] = int((multi > 1).sum())
        f["visits_dropped"] = int(d["dropped"].sum())
    f["ships_listed"] = len(imos)
    f["ships_in_rerun"] = int(rerun["imo"].nunique())
    emit(res, "identity_rerun", f)
    pd.set_option("display.width", 200)
    print(t[t["scope"] == "own"][["set", "main_n_full", "all_ids_n_full", "main_touch_w",
                                  "all_ids_touch_w", "main_feasible_w", "all_ids_feasible_w"]]
          .round(3).to_string(index=False))
    print(yt.pivot_table(index="year", columns="version", values="n_full").to_string())
    for k, v in f.items():
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
