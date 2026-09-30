from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPLETE, HEAD, TAIL, UNANCHORED = 0, 1, 2, 3
KIND_NAMES = {COMPLETE: "complete", HEAD: "head", TAIL: "tail", UNANCHORED: "unanchored"}


@dataclass
class Prepared:
    imo: np.ndarray
    vessel: np.ndarray
    node: np.ndarray
    node_labels: np.ndarray
    time: np.ndarray
    year: np.ndarray
    leg_nm: np.ndarray
    leg_ok: np.ndarray
    new_vessel: np.ndarray
    new_run: np.ndarray
    run: np.ndarray
    run_first: np.ndarray
    run_last: np.ndarray
    D: np.ndarray
    W: np.ndarray
    leg_w: np.ndarray
    weighted: bool
    src: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.node)

    @property
    def n_nodes(self) -> int:
        return len(self.node_labels)


def prepare(stops: pd.DataFrame, node_labels, t0=None, t1=None,
            weight_col: str | None = None) -> Prepared:
    cols = ["imo", "visit_start", "node", "leg_nm", "leg_ok"]
    if weight_col:
        cols.append(weight_col)
    s = stops[cols]
    t = pd.to_datetime(s["visit_start"], utc=True).dt.tz_localize(None)
    if t0 is not None or t1 is not None:
        keep = np.ones(len(s), bool)
        if t0 is not None:
            keep &= (t >= pd.Timestamp(t0)).to_numpy()
        if t1 is not None:
            keep &= (t < pd.Timestamp(t1)).to_numpy()
        s, t = s[keep], t[keep]
    order = np.lexsort((t.to_numpy(), s["imo"].to_numpy().astype(str)))
    s = s.iloc[order]
    t = t.iloc[order]

    n = len(s)
    imo = s["imo"].to_numpy().astype(str)
    vessel = pd.factorize(imo)[0] if n else np.zeros(0, np.int64)
    labels = np.asarray(node_labels, dtype=object)
    if len(set(labels)) != len(labels):
        raise ValueError("node_labels must be unique")
    node = pd.Index(labels).get_indexer(s["node"]).astype(np.int64)

    new_vessel = np.ones(n, bool)
    if n:
        new_vessel[1:] = vessel[1:] != vessel[:-1]
    leg_ok = s["leg_ok"].to_numpy().astype(bool)
    leg = s["leg_nm"].to_numpy(float)
    new_run = new_vessel | ~leg_ok | np.isnan(leg)
    inc = np.where(new_run, 0.0, leg)
    if weight_col:
        w = s[weight_col].to_numpy(float)
        incw = np.where(new_run | np.isnan(w), 0.0, w)
    else:
        incw = inc

    idx = np.arange(n)
    run = np.cumsum(new_run) - 1
    run_first = np.maximum.accumulate(np.where(new_run, idx, 0)) if n else idx
    ends = np.ones(n, bool)
    if n:
        ends[:-1] = new_run[1:]
    run_last = (np.minimum.accumulate(np.where(ends, idx, n)[::-1])[::-1]
                if n else idx)
    cs, csw = np.cumsum(inc), np.cumsum(incw)
    D = cs - cs[run_first] if n else cs
    W = csw - csw[run_first] if n else csw

    tt = t.to_numpy().astype("datetime64[ns]")
    year = pd.DatetimeIndex(tt).year.to_numpy() if n else np.zeros(0, int)
    return Prepared(imo=imo, vessel=vessel, node=node, node_labels=labels,
                    time=tt, year=year, leg_nm=inc, leg_ok=leg_ok,
                    new_vessel=new_vessel, new_run=new_run, run=run,
                    run_first=run_first, run_last=run_last, D=D, W=W,
                    leg_w=incw, weighted=bool(weight_col), src=s.index.to_numpy())


def last_refuel(prep: Prepared, refuel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = prep.n
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(refuel, idx, -1)) if n else idx
    prev = np.maximum(idx - 1, 0)
    lr = last[prev] if n else idx
    anchored = (lr >= prep.run_first) & ~prep.new_run
    return np.where(anchored, lr, -1), anchored


def leg_start(prep: Prepared, refuel: np.ndarray, by: str = "distance"
              ) -> tuple[np.ndarray, np.ndarray]:
    n = prep.n
    if n == 0:
        return np.zeros(0), np.zeros(0, bool)
    lr, anchored = last_refuel(prep, refuel)
    prev = np.maximum(np.arange(n) - 1, 0)
    C = prep.D if by == "distance" else prep.W
    d0 = np.where(anchored, C[prev] - C[np.maximum(lr, 0)], C[prev])
    d0[prep.new_run] = 0.0
    return d0, anchored


def fraction_within(prep: Prepared, d0: np.ndarray, anchored: np.ndarray, E,
                    anchored_only: bool = True, by: str = "distance") -> np.ndarray:
    n = prep.n
    E = np.broadcast_to(np.asarray(E, float), (n,))
    L = prep.leg_nm if by == "distance" else prep.leg_w
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(L > 0, np.clip((E - d0) / L, 0.0, 1.0), (d0 <= E).astype(float))
    f = np.where(np.isnan(E), 0.0, f)
    f[prep.new_run] = 0.0
    if anchored_only:
        f[~anchored] = 0.0
    return f


def leg_fraction(prep: Prepared, refuel: np.ndarray, E, anchored_only: bool = True
                 ) -> np.ndarray:
    if prep.n == 0:
        return np.zeros(0)
    d0, anchored = leg_start(prep, refuel)
    return fraction_within(prep, d0, anchored, E, anchored_only)


def allocate_by_priority(prep: Prepared, refuel: np.ndarray, capacity, priority: np.ndarray,
                         anchored_only: bool = True) -> np.ndarray:
    n = prep.n
    f = np.zeros(n)
    if n == 0:
        return f
    lr, anchored = last_refuel(prep, refuel)
    seg = np.where(anchored, lr, -1 if anchored_only else prep.run_first)
    seg[prep.new_run] = -1
    legs = np.flatnonzero(seg >= 0)
    if len(legs) == 0:
        return f
    cap = np.broadcast_to(np.asarray(capacity, float), (n,))
    sid = seg[legs]
    order = np.lexsort((legs, -priority[legs], sid))
    L, S = legs[order], sid[order]
    w = prep.leg_w[L]
    cw = np.cumsum(w)
    first = np.r_[True, S[1:] != S[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(len(L)), 0))
    before = cw - w - (cw[start] - w[start])
    C = cap[S]
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = np.where(w > 0, np.clip((C - before) / w, 0.0, 1.0), (before < C).astype(float))
    f[L] = np.where(np.isnan(C), 0.0, fr)
    return f


def refuel_mask(prep: Prepared, node_ids) -> np.ndarray:
    want = pd.Index(prep.node_labels).get_indexer(pd.Index(list(node_ids)))
    want = want[want >= 0]
    return np.isin(prep.node, want)


def segments(prep: Prepared, refuel: np.ndarray) -> pd.DataFrame:
    n = prep.n
    D, W, run, time = prep.D, prep.W, prep.run, prep.time
    parts = []

    def add(kind, left, right, length, weight):
        if len(left):
            parts.append(pd.DataFrame({
                "vessel": prep.vessel[left], "run": run[left], "kind": kind,
                "left": left, "right": right, "length_nm": length,
                "weight": weight}))

    r = np.flatnonzero(refuel)
    if len(r) > 1:
        a, b = r[:-1], r[1:]
        same = run[a] == run[b]
        a, b = a[same], b[same]
        add(COMPLETE, a, b, D[b] - D[a], W[b] - W[a])

    starts = np.flatnonzero(prep.new_run)
    lasts = prep.run_last[starts]
    if len(r):
        k = np.searchsorted(r, starts)
        kk = np.minimum(k, len(r) - 1)
        has = (k < len(r)) & (r[kk] <= lasts)
    else:
        kk = np.zeros(len(starts), np.int64)
        has = np.zeros(len(starts), bool)
    f = r[kk[has]] if len(r) else np.zeros(0, np.int64)
    rs, rl = starts[has], lasts[has]
    h = f > rs
    add(HEAD, rs[h], f[h], D[f[h]], W[f[h]])
    if len(r):
        kl = np.searchsorted(r, lasts[has], side="right") - 1
        l = r[kl]
        tl = l < rl
        add(TAIL, l[tl], rl[tl], D[rl[tl]] - D[l[tl]], W[rl[tl]] - W[l[tl]])
    un = ~has & (lasts > starts)
    add(UNANCHORED, starts[un], lasts[un], D[lasts[un]], W[lasts[un]])

    if not parts:
        return pd.DataFrame({"vessel": np.zeros(0, np.int64), "run": np.zeros(0, np.int64),
                             "kind": np.zeros(0, np.int64), "left": np.zeros(0, np.int64),
                             "right": np.zeros(0, np.int64), "length_nm": np.zeros(0),
                             "weight": np.zeros(0)})
    seg = pd.concat(parts, ignore_index=True)
    return seg.sort_values(["vessel", "left", "kind"], kind="mergesort").reset_index(drop=True)


def ship_years(prep: Prepared, seg: pd.DataFrame, refuel: np.ndarray,
               ranges=()) -> pd.DataFrame:
    yr = prep.year
    st = pd.DataFrame({
        "vessel": prep.vessel, "year": yr,
        "dist_nm": prep.leg_nm,
        "w_total": prep.leg_w,
        "brk": (~prep.leg_ok) & (~prep.new_vessel),
        "touch": refuel,
    })
    g = st.groupby(["vessel", "year"], sort=True)
    out = pd.DataFrame({
        "n_stops": g.size(),
        "dist_nm": g["dist_nm"].sum(),
        "w_total": g["w_total"].sum(),
        "n_breaks": g["brk"].sum().astype(int),
        "touches": g["touch"].any(),
        "n_refuel_stops": g["touch"].sum().astype(int),
    })

    if len(seg):
        y0 = yr[seg["left"].to_numpy()]
        y1 = yr[seg["right"].to_numpy()]
        reps = (y1 - y0 + 1).astype(int)
        ex = seg.loc[seg.index.repeat(reps), ["vessel", "kind", "length_nm"]].copy()
        offs = np.arange(int(reps.sum())) - np.repeat(np.cumsum(reps) - reps, reps)
        ex["year"] = np.repeat(y0, reps) + offs
        ex["censored"] = (ex["kind"] != COMPLETE).to_numpy()
        comp = ex[ex["kind"] == COMPLETE]
        gm = ex.groupby(["vessel", "year"])
        out = out.join(pd.DataFrame({
            "req_lb": gm["length_nm"].max(),
            "n_segments": gm.size(),
            "n_censored": gm["censored"].sum(),
        }), how="left")
        out = out.join(comp.groupby(["vessel", "year"])["length_nm"].max()
                       .rename("req_complete"), how="left")
        if len(ranges):
            c = seg[seg["kind"] == COMPLETE]
            cy = yr[c["right"].to_numpy()]
            L = c["length_nm"].to_numpy()
            for R in ranges:
                ok = L <= R
                cov = (pd.DataFrame({"vessel": c["vessel"].to_numpy(), "year": cy,
                                     "d": np.where(ok, L, 0.0),
                                     "w": np.where(ok, c["weight"].to_numpy(), 0.0)})
                       .groupby(["vessel", "year"])[["d", "w"]].sum()
                       .reindex(out.index).fillna(0.0))
                out[f"covered_{int(R)}"] = cov["d"]
                if prep.weighted:
                    out[f"covered_w_{int(R)}"] = cov["w"]
    else:
        out["req_lb"] = np.nan
        out["n_segments"] = 0
        out["n_censored"] = 0
        out["req_complete"] = np.nan
        for R in ranges:
            out[f"covered_{int(R)}"] = 0.0
            if prep.weighted:
                out[f"covered_w_{int(R)}"] = 0.0

    out["n_segments"] = out["n_segments"].fillna(0).astype(int)
    out["n_censored"] = out["n_censored"].fillna(0).astype(int)
    out = out.reset_index()
    labels = pd.Series(prep.imo).groupby(prep.vessel).first()
    out.insert(0, "imo", out["vessel"].map(labels))
    return out


def objective(prep: Prepared, refuel: np.ndarray, R: float) -> float:
    r = np.flatnonzero(refuel)
    if len(r) < 2:
        return 0.0
    a, b = r[:-1], r[1:]
    same = prep.run[a] == prep.run[b]
    L = prep.D[b] - prep.D[a]
    w = prep.W[b] - prep.W[a]
    return float(np.sum(np.where(same & (L <= R), w, 0.0)))


def total_weight(prep: Prepared) -> float:
    if prep.n == 0:
        return 0.0
    return float(np.sum(prep.W[np.flatnonzero(np.r_[prep.new_run[1:], True])]))


def node_order(prep: Prepared) -> np.ndarray:
    return np.lexsort((np.arange(prep.n), prep.node, prep.run))


def marginal_gains(prep: Prepared, refuel: np.ndarray, R: float,
                   order: np.ndarray | None = None) -> np.ndarray:
    n = prep.n
    gains = np.zeros(prep.n_nodes)
    if n == 0:
        return gains
    if order is None:
        order = node_order(prep)
    idx = np.arange(n)
    rf, rl, D, W = prep.run_first, prep.run_last, prep.D, prep.W

    last_ref = np.maximum.accumulate(np.where(refuel, idx, -1))
    has_left = last_ref >= rf
    next_ref = np.minimum.accumulate(np.where(refuel, idx, n)[::-1])[::-1]
    has_right = next_ref <= rl
    li = np.clip(last_ref, 0, n - 1)
    ri = np.clip(next_ref, 0, n - 1)
    D_left = np.where(has_left, D[li], 0.0)
    W_left = np.where(has_left, W[li], 0.0)
    D_right = np.where(has_right, D[ri], D[rl])
    W_right = np.where(has_right, W[ri], W[rl])
    piece = np.where(has_left, last_ref, n + prep.run)
    covered = has_left & has_right & (D_right - D_left <= R)

    eligible = ~refuel & ~covered & (prep.node >= 0)
    sel = order[eligible[order]]
    if sel.size == 0:
        return gains
    p, nd, d, w = piece[sel], prep.node[sel], D[sel], W[sel]
    same_prev = np.zeros(sel.size, bool)
    same_prev[1:] = (p[1:] == p[:-1]) & (nd[1:] == nd[:-1])
    same_next = np.zeros(sel.size, bool)
    same_next[:-1] = same_prev[1:]
    d_prev = np.r_[0.0, d[:-1]]
    w_prev = np.r_[0.0, w[:-1]]

    len_a = np.where(same_prev, d - d_prev, d - D_left[sel])
    w_a = np.where(same_prev, w - w_prev, w - W_left[sel])
    comp_a = same_prev | has_left[sel]
    contrib = np.where(comp_a & (len_a <= R), w_a, 0.0)

    len_b = D_right[sel] - d
    w_b = W_right[sel] - w
    comp_b = ~same_next & has_right[sel]
    contrib = contrib + np.where(comp_b & (len_b <= R), w_b, 0.0)
    return np.bincount(nd, weights=contrib, minlength=prep.n_nodes)
