from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chains
from common import (fork_pool, DEFAULT_JOBS, LEG_OK, parse_floats, parse_years, require,
                    tagged, to_utc)
from facts import emit

HERE = Path(__file__).resolve().parent
HUBS = HERE / "reference" / "hub_sets.csv"

SET_COLS = ["set_key", "method", "train_year", "sel_range_nm", "groups", "rank",
            "node_id", "node_name", "node_iso3", "train_visits", "gain",
            "cum_share", "fallback", "hub", "initial"]


def set_key(method, year, R, groups) -> str:
    return f"{method}|{int(year)}|{int(R)}|{groups}"


def load_inputs(out_dir: Path, stops_name="stops.csv.gz", nodes_name="nodes.csv",
                vessels_name="vessels.csv", groups=None, slow_as_break=False,
                extra=()):
    require(out_dir, stops_name, nodes_name, *([vessels_name] if groups else []))
    cols = ["imo", "visit_start", "anchorage_id", "leg_nm", "leg_status"]
    cols += [c for c in (["slow"] if slow_as_break else []) + list(extra) if c not in cols]
    stops = pd.read_csv(out_dir / stops_name, dtype=str, low_memory=False, usecols=cols)
    stops["visit_start"] = to_utc(stops["visit_start"])
    stops["leg_nm"] = pd.to_numeric(stops["leg_nm"], errors="coerce")
    if "sea_hours" in stops.columns:
        stops["sea_hours"] = pd.to_numeric(stops["sea_hours"], errors="coerce")
    stops["leg_ok"] = (stops["leg_status"] == LEG_OK).to_numpy()
    if slow_as_break:
        slow = (stops["slow"] == "True").to_numpy()
        print(f"  slow-as-break: {int((slow & stops['leg_ok']).sum()):,} slow legs "
              f"become breaks")
        stops["leg_ok"] = stops["leg_ok"].to_numpy() & ~slow
    nodes = pd.read_csv(out_dir / nodes_name, dtype={"anchorage_id": str, "node_id": str})
    stops["node"] = stops["anchorage_id"].map(nodes.set_index("anchorage_id")["node_id"])
    if groups:
        v = pd.read_csv(out_dir / vessels_name, dtype={"imo": str})
        keep = set(v.loc[v["group"].isin(groups), "imo"])
        stops = stops[stops["imo"].isin(keep)].reset_index(drop=True)
    labels = np.array(sorted(nodes["node_id"].dropna().unique()), dtype=object)
    return stops, nodes, labels


def year_window(year: int):
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")


def node_volume(prep: chains.Prepared) -> np.ndarray:
    ok = prep.node >= 0
    return np.bincount(prep.node[ok], minlength=prep.n_nodes).astype(float)


def rank_by_volume(vol: np.ndarray, nmax: int) -> list[dict]:
    codes = np.lexsort((np.arange(len(vol)), -vol))
    codes = [c for c in codes if vol[c] > 0][:nmax]
    return [{"rank": i + 1, "code": int(c), "gain": float(vol[c]),
             "fallback": False} for i, c in enumerate(codes)]


def greedy(prep: chains.Prepared, R: float, nmax: int, candidates: np.ndarray,
           vol: np.ndarray, initial=()) -> list[dict]:
    order = chains.node_order(prep)
    total = chains.total_weight(prep) or 1.0
    chosen = [int(c) for c in initial]
    refuel = np.isin(prep.node, chosen) if chosen else np.zeros(prep.n, bool)
    obj = chains.objective(prep, refuel, R)
    is_cand = np.zeros(prep.n_nodes, bool)
    is_cand[np.asarray(candidates, int)] = True
    is_cand[chosen] = False
    out = []
    for k in range(nmax):
        if not is_cand.any():
            break
        g = chains.marginal_gains(prep, refuel, R, order)
        g = np.where(is_cand, g, -np.inf)
        key = np.lexsort((np.arange(len(g)), -vol, -np.round(g, 6)))
        best = int(key[0])
        fallback = not g[best] > 0
        if fallback:
            cand_codes = np.flatnonzero(is_cand)
            best = int(cand_codes[np.lexsort((cand_codes, -vol[cand_codes]))[0]])
        refuel |= prep.node == best
        new_obj = chains.objective(prep, refuel, R)
        out.append({"rank": len(chosen) + k + 1, "code": best,
                    "gain": new_obj - obj, "cum_share": new_obj / total,
                    "fallback": bool(fallback)})
        obj = new_obj
        is_cand[best] = False
    return out


def hub_nodes(hubs: pd.DataFrame, nodes: pd.DataFrame):
    node_of = nodes.set_index("anchorage_id")["node_id"]
    out: dict[str, list] = {}
    for set_id, grp in hubs.groupby("set_id", sort=False):
        items = []
        for hub, ids in grp.groupby("hub", sort=False):
            a = ids["anchorage_id"].astype(str).tolist()
            found = [node_of[x] for x in a if x in node_of.index]
            items.append((hub, sorted(set(found)), [x for x in a if x not in node_of.index]))
        out[set_id] = items
    return out


def external_frame(hubs: pd.DataFrame, nodes: pd.DataFrame, labels, gtag="all") -> pd.DataFrame:
    code_of = {nid: i for i, nid in enumerate(labels)}
    frames = []
    for set_id, items in hub_nodes(hubs, nodes).items():
        rank = 0
        for hub, nids, missing in items:
            print(f"  external {set_id} {hub}: {', '.join(nids) or 'NO NODE MATCHED'}"
                  + (f"   (not in data: {', '.join(missing)})" if missing else ""))
            for nid in nids:
                rank += 1
                frames.append(rows_to_frame([{"rank": rank, "code": code_of[nid]}], labels,
                                            nodes, None, f"external:{set_id}", -1, 0, gtag, hub))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SET_COLS)


def read_hubs(path: Path = HUBS) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["set_id", "hub", "anchorage_id", "source"])
    return pd.read_csv(path, comment="#", dtype=str)


def read_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {l.strip() for l in Path(path).read_text().split() if l.strip()}


_STOPS = None
_LABELS = None
_EXCLUDED = np.zeros(0, bool)


def candidate_codes(vol: np.ndarray, ncand: int, excluded: np.ndarray) -> list[int]:
    order = np.lexsort((np.arange(len(vol)), -vol))
    ok = (vol > 0) & ~(excluded if len(excluded) else np.zeros(len(vol), bool))
    return [int(c) for c in order if ok[c]][:ncand]


def _greedy_job(args):
    year, R, nmax, ncand, initial_ids = args
    t0, t1 = year_window(year)
    prep = chains.prepare(_STOPS, _LABELS, t0, t1)
    vol = node_volume(prep)
    cand = candidate_codes(vol, ncand, _EXCLUDED)
    init = [int(c) for c in pd.Index(_LABELS).get_indexer(initial_ids) if c >= 0]
    rows = greedy(prep, R, nmax, np.asarray(cand, int), vol, init)
    return year, R, rows, vol, init


def rows_to_frame(rows, labels, nodes, vol, method, year, R, groups, hub=None,
                  initial=False):
    lead = nodes.drop_duplicates("node_id").set_index("node_id")
    out = []
    for r in rows:
        nid = labels[r["code"]]
        out.append({
            "set_key": set_key(method, year, R, groups), "method": method,
            "train_year": int(year), "sel_range_nm": int(R), "groups": groups,
            "rank": r["rank"], "node_id": nid,
            "node_name": lead["node_name"].get(nid, ""),
            "node_iso3": lead["node_iso3"].get(nid, ""),
            "train_visits": float(vol[r["code"]]) if vol is not None else np.nan,
            "gain": r.get("gain", np.nan), "cum_share": r.get("cum_share", np.nan),
            "fallback": r.get("fallback", False), "hub": hub or "",
            "initial": bool(initial)})
    return pd.DataFrame(out, columns=SET_COLS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--years", required=True, help="e.g. 2018-2025 or 2019,2022")
    ap.add_argument("--methods", default="volume,greedy")
    ap.add_argument("--ranges", default="7000", help="endurance(s) for greedy, nm")
    ap.add_argument("--nmax", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=1500)
    ap.add_argument("--groups", default="", help="restrict to vessel groups")
    ap.add_argument("--hubs", type=Path, default=HUBS)
    ap.add_argument("--exclude-candidates", type=Path, default=None,
                    help="file of node ids that may not be chosen (one per line)")
    ap.add_argument("--initial-key", default="",
                    help="set_key in an existing port_sets.csv to extend (backup nodes)")
    ap.add_argument("--initial-n", type=int, default=0)
    ap.add_argument("--append", action="store_true",
                    help="add to an existing port_sets.csv instead of overwriting")
    ap.add_argument("--stops", default="stops.csv.gz")
    ap.add_argument("--nodes", default="nodes.csv")
    ap.add_argument("--slow-as-break", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix for outputs, e.g. km15 -> port_sets_km15.csv")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--out", default="port_sets.csv")
    a = ap.parse_args(argv)
    a.out = tagged(a.out, a.tag)
    out_dir = a.out_dir.expanduser().resolve()
    years = parse_years(a.years)
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    ranges = parse_floats(a.ranges)
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    gtag = "+".join(groups) if groups else "all"

    stops, nodes, labels = load_inputs(out_dir, a.stops, a.nodes, groups=groups,
                                       slow_as_break=a.slow_as_break)
    excluded_ids = read_exclusions(a.exclude_candidates)
    excluded = np.isin(labels, list(excluded_ids))
    if excluded_ids:
        print(f"  {int(excluded.sum())} of {len(excluded_ids)} excluded ids are nodes here; "
              f"they cannot be chosen")
    print(f"{len(stops):,} stops, {stops['imo'].nunique():,} vessels, "
          f"{len(labels):,} nodes; groups={gtag}")
    frames = []

    initial_ids, method_greedy = [], "greedy"
    if a.initial_key:
        require(out_dir, a.out, stage="select_ports.py without --initial-key")
        prev = pd.read_csv(out_dir / a.out, dtype={"node_id": str})
        base = prev[(prev["set_key"] == a.initial_key) & (prev["rank"] <= a.initial_n)]
        if base.empty:
            sys.exit(f"--initial-key {a.initial_key} n<={a.initial_n}: no such set")
        initial_ids = base.sort_values("rank")["node_id"].tolist()
        method_greedy = f"backup[{a.initial_key.replace('|', ':')}@{a.initial_n}]"
        methods = ["greedy"]
        print(f"extending {len(initial_ids)} nodes of {a.initial_key}")

    if "volume" in methods:
        for y in years:
            t0, t1 = year_window(y)
            prep = chains.prepare(stops, labels, t0, t1)
            vol = node_volume(prep)
            ranked = rank_by_volume(np.where(excluded, 0.0, vol), a.nmax)
            frames.append(rows_to_frame(ranked, labels, nodes, vol, "volume", y, 0, gtag))
            print(f"  volume {y}: {int((vol > 0).sum()):,} nodes visited")

    if "greedy" in methods:
        global _STOPS, _LABELS, _EXCLUDED
        _STOPS, _LABELS, _EXCLUDED = stops, labels, excluded
        jobs = [(y, R, a.nmax, a.candidates, initial_ids) for y in years for R in ranges]
        n = max(1, min(a.jobs, len(jobs)))
        print(f"greedy: {len(jobs)} (year, range) job(s) on {n} process(es)")
        pool = fork_pool(n)
        if pool is not None:
            with pool:
                results = pool.map(_greedy_job, jobs)
        else:
            results = [_greedy_job(j) for j in jobs]
        for y, R, rows, vol, init in results:
            fb = sum(r["fallback"] for r in rows)
            share = rows[-1]["cum_share"] if rows else 0.0
            print(f"  greedy {y} R={R:g}: {len(rows)} added, covered share "
                  f"{share:.3f}, {fb} fallback(s)")
            if init:
                frames.append(rows_to_frame(
                    [{"rank": i + 1, "code": c} for i, c in enumerate(init)],
                    labels, nodes, vol, method_greedy, y, R, gtag, initial=True))
            frames.append(rows_to_frame(rows, labels, nodes, vol, method_greedy,
                                        y, R, gtag))
        _STOPS = None

    if not a.initial_key and not a.append:
        ext = external_frame(read_hubs(a.hubs), nodes, labels, gtag)
        if len(ext):
            frames.append(ext)

    out = out_dir / a.out
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SET_COLS)
    if (a.initial_key or a.append) and out.exists():
        old = pd.read_csv(out, dtype={"node_id": str})
        old = old[~old["set_key"].isin(set(new["set_key"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)
    tag = "_backup" if a.initial_key else (f"_{gtag}" if a.append else "")
    emit(out_dir, tagged("select_ports" + tag, a.tag), {
        "sets": int(new["set_key"].nunique()), "years": ",".join(map(str, years)),
        "groups": gtag, "nmax": a.nmax, "candidates": a.candidates})
    print(f"wrote {out} ({new['set_key'].nunique()} sets)")


if __name__ == "__main__":
    main()
