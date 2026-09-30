from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
from common import fork_pool, DEFAULT_JOBS, parse_years
from facts import emit
from select_ports import (SET_COLS, candidate_codes, load_inputs, node_volume,
                          rows_to_frame, year_window)

_PREP: chains.Prepared | None = None
_BASE: np.ndarray | None = None
_R: float = 7000.0


def objective(prep: chains.Prepared, refuel: np.ndarray, R: float) -> float:
    return float((chains.leg_fraction(prep, refuel, R) * prep.leg_nm).sum())


def _gain(code: int) -> float:
    r = _BASE | (_PREP.node == code)
    return objective(_PREP, r, _R)


def greedy_partial(prep, R, nmax, candidates, vol, jobs=1) -> list[dict]:
    global _PREP, _BASE, _R
    _PREP, _R = prep, R
    total = float(prep.leg_nm.sum()) or 1.0
    refuel = np.zeros(prep.n, bool)
    obj = 0.0
    cand = [int(c) for c in candidates]
    out = []
    for k in range(nmax):
        if not cand:
            break
        _BASE = refuel
        pool = fork_pool(jobs)
        if pool is not None:
            with pool:
                vals = pool.map(_gain, cand, chunksize=max(1, len(cand) // (4 * jobs)))
        else:
            vals = [_gain(c) for c in cand]
        g = np.asarray(vals) - obj
        ca = np.asarray(cand)
        order = np.lexsort((ca, -vol[ca], -np.round(g, 6)))
        best_i = int(order[0])
        fallback = not g[best_i] > 0
        if fallback:
            best_i = int(np.lexsort((ca, -vol[ca]))[0])
        best = int(ca[best_i])
        refuel = refuel | (prep.node == best)
        new_obj = objective(prep, refuel, R)
        out.append({"rank": k + 1, "code": best, "gain": new_obj - obj,
                    "cum_share": new_obj / total, "fallback": bool(fallback)})
        obj = new_obj
        cand.remove(best)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--years", required=True)
    ap.add_argument("--range", dest="R", type=float, default=7000.0)
    ap.add_argument("--nmax", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=1500)
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--out", default="port_sets_dualfuel.csv")
    a = ap.parse_args(argv)
    out_dir = a.out_dir.expanduser().resolve()
    stops, nodes, labels = load_inputs(out_dir)
    frames = []
    for y in parse_years(a.years):
        t0, t1 = year_window(y)
        prep = chains.prepare(stops, labels, t0, t1)
        vol = node_volume(prep)
        cand = candidate_codes(vol, a.candidates, np.zeros(len(vol), bool))
        rows = greedy_partial(prep, a.R, a.nmax, cand, vol, a.jobs)
        frames.append(rows_to_frame(rows, labels, nodes, vol, "dualfuel", y, a.R, "all"))
        fb = sum(r["fallback"] for r in rows)
        print(f"  dualfuel {y} R={a.R:g}: {len(rows)} added, partial share "
              f"{rows[-1]['cum_share'] if rows else 0:.3f}, {fb} fallback(s)")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SET_COLS)
    out.to_csv(out_dir / a.out, index=False)
    emit(out_dir, "select_dualfuel", {"years": a.years, "range_nm": a.R, "nmax": a.nmax,
                                      "candidates": a.candidates,
                                      "sets": int(out["set_key"].nunique())})
    print(f"-> {out_dir / a.out}")


if __name__ == "__main__":
    main()
