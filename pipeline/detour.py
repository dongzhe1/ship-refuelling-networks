from __future__ import annotations

import numpy as np
import pandas as pd

import chains

EARTH_NM = 3440.065


def gc_nm(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_NM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def node_coords(nodes: pd.DataFrame, labels) -> tuple[np.ndarray, np.ndarray]:
    lead = nodes.drop_duplicates("node_id").set_index("node_id")
    lat = pd.Series(list(labels)).map(lead["node_lat"]).to_numpy(float)
    lon = pd.Series(list(labels)).map(lead["node_lon"]).to_numpy(float)
    return lat, lon


def leg_detour(prep: chains.Prepared, lat: np.ndarray, lon: np.ndarray,
               supply_codes) -> np.ndarray:
    n = prep.n
    out = np.full(n, np.inf)
    sup = np.asarray(sorted(set(int(c) for c in supply_codes)), int)
    if n < 2 or len(sup) == 0:
        return out
    a = np.r_[-1, prep.node[:-1]]
    b = prep.node
    ok = ~prep.new_run & (a >= 0) & (b >= 0)
    pairs = np.unique(np.stack([a[ok], b[ok]], axis=1), axis=0)
    if len(pairs) == 0:
        return out
    la, loa = lat[pairs[:, 0]], lon[pairs[:, 0]]
    lb, lob = lat[pairs[:, 1]], lon[pairs[:, 1]]
    direct = gc_nm(la, loa, lb, lob)
    best = np.full(len(pairs), np.inf)
    for s in sup:
        d = gc_nm(la, loa, lat[s], lon[s]) + gc_nm(lat[s], lon[s], lb, lob) - direct
        best = np.fmin(best, d)
    best = np.where(np.isnan(best), np.inf, best)
    key_pairs = pairs[:, 0].astype(np.int64) * (prep.n_nodes + 1) + pairs[:, 1]
    key_legs = a[ok].astype(np.int64) * (prep.n_nodes + 1) + b[ok]
    out[np.flatnonzero(ok)] = best[np.searchsorted(key_pairs, key_legs)]
    return out


def insert_refuels(prep: chains.Prepared, refuel: np.ndarray, E, detour: np.ndarray,
                   delta: float) -> tuple[np.ndarray, dict]:
    n = prep.n
    aug = refuel.copy()
    stats = {"segments": 0, "inserted": 0, "extra_nm": 0.0, "chosen": np.zeros(0, np.int64)}
    if n == 0 or delta <= 0:
        return aug, stats
    idx = np.arange(n)
    lr, anchored = chains.last_refuel(prep, refuel)
    start = np.where(anchored, lr, prep.run_first)
    nxt = np.minimum.accumulate(np.where(refuel, idx, n)[::-1])[::-1]
    end = np.minimum(nxt, prep.run_last)
    D = prep.D
    legs = ~prep.new_run
    x = D[np.maximum(idx - 1, 0)] - D[start]
    L = D[end] - D[start]
    Es = np.broadcast_to(np.asarray(E, float), (n,))[start]
    with np.errstate(invalid="ignore"):
        gain = np.where(anchored,
                        np.minimum(x, Es) + np.minimum(L - x, Es) - np.minimum(L, Es),
                        np.minimum(L - x, Es))
    ok = legs & np.isfinite(gain) & (gain > 1e-9) & (detour <= delta * L)
    stats["segments"] = int(len(np.unique(start[legs])))
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return aug, stats
    order = np.lexsort((cand, -gain[cand], start[cand]))
    c = cand[order]
    first = np.r_[True, start[c][1:] != start[c][:-1]]
    chosen = c[first]
    aug[chosen - 1] = True
    stats["inserted"] = int(len(chosen))
    stats["extra_nm"] = float(detour[chosen].sum())
    stats["chosen"] = chosen
    return aug, stats
