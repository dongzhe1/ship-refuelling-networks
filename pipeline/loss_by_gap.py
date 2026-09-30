from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import parse_years, require
from facts import emit
from pilot import wilson

GROUPS = ("all", "container", "bulk", "tanker")


def read_greedy(out_dir: Path, n: int, R: int) -> pd.DataFrame:
    cols = ["set_key", "n_ports", "imo", "year", "vessel_group", "req_lb", "n_breaks"]
    parts = []
    for ch in pd.read_csv(out_dir / "ship_years.csv.gz", usecols=cols, dtype={"imo": str},
                          chunksize=2_000_000):
        ch = ch[ch["set_key"].str.startswith("greedy|") & ch["set_key"].str.endswith(f"|{R}|all")
                & (ch["n_ports"] == n) & (ch["n_breaks"] == 0) & ch["req_lb"].notna()]
        if len(ch):
            parts.append(ch)
    d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    d["train_year"] = d["set_key"].str.split("|").str[1].astype(int)
    return d


def strata(vessels: pd.DataFrame) -> pd.Series:
    v = vessels[["imo", "group"]].copy()
    gt = pd.to_numeric(vessels.get("gt", pd.Series(np.nan, index=vessels.index)), errors="coerce")
    v["t"] = gt.groupby(v["group"]).transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) if s.notna().sum() >= 3 else 0)
    v["t"] = v["t"].fillna(-1).astype(int)
    return pd.Series((v["group"] + ":" + v["t"].astype(str)).to_numpy(), index=v["imo"].astype(str))


TRANSITION_COLS = ["train_year", "eval_year", "gap", "vessel_group", "n", "lost",
                   "lost_share", "lo", "hi", "indep_share"]


def transitions(d: pd.DataFrame, stratum: pd.Series, R: int, years) -> pd.DataFrame:
    rows = []
    for t in sorted(d["train_year"].unique()):
        dt_ = d[d["train_year"] == t]
        base = dt_[dt_["year"] == t].set_index("imo")
        for y in years:
            if y <= t:
                continue
            ev = dt_[dt_["year"] == y].set_index("imo")
            j = base[["req_lb", "vessel_group"]].join(ev[["req_lb"]], lsuffix="_t", rsuffix="_y",
                                                      how="inner")
            pop = j[j["req_lb_t"] <= R].copy()
            ev = ev.assign(stratum=ev.index.map(stratum).fillna("unknown"),
                           lost=(ev["req_lb"] > R).astype(float))
            f_not = ev.groupby("stratum")["lost"].mean()
            pop["stratum"] = pop.index.map(stratum).fillna("unknown")
            pop["indep"] = pop["stratum"].map(f_not)
            pop["lost"] = pop["req_lb_y"] > R
            for g in GROUPS:
                p = pop if g == "all" else pop[pop["vessel_group"] == g]
                n = len(p)
                if n == 0:
                    continue
                share = float(p["lost"].mean())
                lo, hi = wilson(share, n)
                rows.append({"train_year": int(t), "eval_year": int(y), "gap": int(y - t),
                             "vessel_group": g, "n": n, "lost": int(p["lost"].sum()),
                             "lost_share": share, "lo": lo, "hi": hi,
                             "indep_share": float(p["indep"].mean())})
    return pd.DataFrame(rows, columns=TRANSITION_COLS)


def summary(tr: pd.DataFrame, train=2019, eval_=2024) -> dict:
    f = {}
    a = tr[tr["vessel_group"] == "all"]
    if not len(a):
        return f
    by_gap = a.groupby("gap").apply(lambda g: g["lost"].sum() / g["n"].sum(), include_groups=False)
    for k, v in by_gap.items():
        f[f"gap{k}_lost_pct"] = 100 * float(v)
    r = a[(a["train_year"] == train) & (a["eval_year"] == eval_)]
    if len(r):
        r = r.iloc[0]
        f.update({"headline_lost_pct": 100 * r["lost_share"], "headline_n": int(r["n"]),
                  "headline_indep_pct": 100 * r["indep_share"]})
        if 1 in by_gap.index:
            f["headline_excess_over_gap1_pp"] = 100 * (r["lost_share"] - by_gap.loc[1])
    one = a[(a["train_year"] == train) & (a["gap"] == 1)]
    if len(one):
        f["train_gap1_lost_pct"] = 100 * float(one["lost_share"].iloc[0])
    return f


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--range", dest="R", type=int, default=7000)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--train", type=int, default=2019)
    ap.add_argument("--eval", dest="eval_", type=int, default=2024)
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    require(out_dir, "ship_years.csv.gz", "vessels.csv", stage="evaluate.py")
    d = read_greedy(out_dir, a.n, a.R)
    vessels = pd.read_csv(out_dir / "vessels.csv", dtype={"imo": str})
    tr = transitions(d, strata(vessels), a.R, parse_years(a.years))
    res = out_dir / "results"
    res.mkdir(exist_ok=True)
    tr.to_csv(res / "loss_by_gap.csv", index=False)
    f = summary(tr, a.train, a.eval_)
    emit(res, "loss_by_gap", {k: v for k, v in f.items() if np.isfinite(v)})
    pd.set_option("display.width", 160)
    if tr.empty:
        print(f"  no ship continuous in one year and fully observed in a later one "
              f"(N={a.n}, R={a.R}): loss_by_gap.csv has no rows")
    else:
        print(tr[tr["vessel_group"] == "all"].pivot_table(index="train_year", columns="gap",
                                                          values="lost_share").round(3).to_string())
    for k, v in f.items():
        print(f"  {k:<32} {v:.2f}" if isinstance(v, float) else f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
