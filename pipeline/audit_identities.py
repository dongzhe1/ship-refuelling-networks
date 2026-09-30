from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pull_port_visits as pv
from facts import emit

N_SAMPLE, SEED = 300, 20260928
START, END = "2018-01-01", "2026-08-01"


def visits(token: str, vid: str, start: str, end: str) -> int | None:
    base = [("datasets[0]", pv.PORTVISIT_DS), ("vessels[0]", vid),
            ("start-date", start), ("end-date", end)]
    d = pv.api_get(token, "events", base + [("limit", "1"), ("offset", "0")])
    if d is None:
        return None
    if isinstance(d.get("total"), int):
        return int(d["total"])
    n, offset = 0, 0
    while True:
        d = pv.api_get(token, "events", base + [("limit", str(pv.PAGE)), ("offset", str(offset))])
        if d is None:
            return None
        entries = d.get("entries") or []
        n += len(entries)
        nxt = d.get("nextOffset")
        if not entries or nxt is None:
            return n
        offset = nxt
        time.sleep(pv.PAUSE)


def read_main_ids(pull: Path) -> dict[str, str]:
    p = pull / "vessel_ids.csv"
    if not p.exists():
        sys.exit(f"{p} not found: point at the main pull's work directory ($GFW)")
    with open(p, newline="", encoding="utf-8") as fh:
        return {r["imo"]: r["vessel_id"] for r in csv.DictReader(fh) if r.get("vessel_id")}


def audit_one(token: str, imo: str, main_id: str, start: str, end: str) -> dict:
    d = pv.api_get(token, "vessels/search", [("query", imo), ("datasets[0]", pv.IDENTITY_DS),
                                             ("limit", str(pv.SEARCH_LIMIT))])
    if d is None:
        return {"imo": imo, "main_id": main_id, "status": "search failed"}
    used, fallback = pv.pick_identities(d.get("entries") or [], imo)
    ids = [si["id"] for si in used]
    win = {si["id"]: (pv._day(si.get("transmissionDateFrom")), pv._day(si.get("transmissionDateTo")))
           for si in used}
    others = [v for v in ids if v != main_id]
    row = {"imo": imo, "main_id": main_id, "n_identities": len(ids),
           "main_among_them": main_id in ids, "fallback": fallback,
           "other_ids": ";".join(others), "status": "ok"}
    row["main_visits"] = visits(token, main_id, start, end)
    extra = [visits(token, v, start, end) for v in others]
    row["other_visits"] = None if any(x is None for x in extra) else sum(extra)
    m0, m1 = win.get(main_id, (None, None))
    s0, s1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    row["main_from"], row["main_to"] = (m0 or ""), (m1 or "")
    if m0 and m1:
        a, b = max(m0, s0), min(m1 + dt.timedelta(days=1), s1)
        row["main_window_share"] = max(0, (b - a).days) / (s1 - s0).days
        inside = []
        for v, n in zip(others, extra):
            if n is None or a >= b:
                inside.append(0 if a >= b else None)
                continue
            inside.append(visits(token, v, a.isoformat(), b.isoformat()) if n else 0)
        if any(x is None for x in inside) or row["other_visits"] is None:
            row["status"] = "incomplete"
        else:
            row["other_inside"] = sum(inside)
            row["other_outside"] = row["other_visits"] - row["other_inside"]
    else:
        row["status"] = "no main window"
    if row["main_visits"] is None or row["other_visits"] is None:
        row["status"] = "incomplete"
    return row


def summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return {"sampled": len(rows), "audited": 0}
    main = sum(r["main_visits"] for r in ok)
    other = sum(r["other_visits"] for r in ok)
    return {
        "sampled": len(rows), "audited": len(ok),
        "multi_identity_pct": 100 * sum(r["n_identities"] > 1 for r in ok) / len(ok),
        "other_identity_with_visits_pct": 100 * sum(r["other_visits"] > 0 for r in ok) / len(ok),
        "main_not_among_identities_pct": 100 * sum(not r["main_among_them"] for r in ok) / len(ok),
        "missing_visit_pct": 100 * other / (main + other) if main + other else 0.0,
        "ships_missing_over_10pct": sum(r["other_visits"] > 0.1 * (r["main_visits"] + r["other_visits"])
                                        for r in ok if r["main_visits"] + r["other_visits"] > 0),
        "sequential_missing_visit_pct": 100 * sum(r["other_outside"] for r in ok)
        / (main + sum(r["other_outside"] for r in ok)) if main else 0.0,
        "concurrent_other_visit_pct": 100 * sum(r["other_inside"] for r in ok) / (main + other)
        if main + other else 0.0,
        "main_window_share_median": float(sorted(r["main_window_share"] for r in ok)[len(ok) // 2]),
        "ships_main_window_under_half": sum(r["main_window_share"] < 0.5 for r in ok),
        "ships_sequential_over_10pct": sum(
            r["other_outside"] > 0.1 * (r["main_visits"] + r["other_outside"])
            for r in ok if r["main_visits"] + r["other_outside"] > 0),
    }


def full_year_shares(audit: pd.DataFrame, years=range(2018, 2026)) -> pd.DataFrame:
    ok = audit[audit["status"] == "ok"]
    t0, t1 = pd.to_datetime(ok["main_from"]), pd.to_datetime(ok["main_to"])
    return pd.DataFrame([{"year": y, "ships": len(ok),
                          "followed_whole_year_share":
                              float(((t0 <= f"{y}-01-01") & (t1 >= f"{y}-12-31")).mean())}
                         for y in years])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("main_pull", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--token-file", type=Path, default=None, help="default ~/.gfw_token")
    a = ap.parse_args(argv)
    pull, out = a.main_pull.expanduser().resolve(), a.out_dir.expanduser().resolve()
    if out == pull or pull in out.parents:
        sys.exit("out_dir must not be inside the main pull directory (read only)")
    out.mkdir(parents=True, exist_ok=True)
    ids = read_main_ids(pull)
    token = pv.read_token(a.token_file)
    sample = random.Random(a.seed).sample(sorted(ids), min(a.n, len(ids)))
    pv.log(f"{len(ids):,} IMOs in the main pull; auditing {len(sample)} (seed {a.seed})")
    path = out / "identity_audit.csv"
    done = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            done = {r["imo"]: r for r in csv.DictReader(fh)}
    cols = ["imo", "main_id", "n_identities", "main_among_them", "fallback", "other_ids",
            "main_visits", "other_visits", "main_from", "main_to", "main_window_share",
            "other_inside", "other_outside", "status"]
    rows = []
    if done and "other_outside" not in next(iter(done.values())):
        sys.exit(f"{path} was written by version 1 of this script; use a new out_dir")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if not done:
            w.writeheader()
        for k, imo in enumerate(sample, 1):
            if imo in done and done[imo]["status"] == "ok" and done[imo].get("other_outside"):
                r = done[imo]
                rows.append({**r, "n_identities": int(r["n_identities"]),
                             "main_among_them": r["main_among_them"] == "True",
                             "main_visits": int(r["main_visits"]),
                             "other_visits": int(r["other_visits"]),
                             "main_window_share": float(r["main_window_share"]),
                             "other_inside": int(r["other_inside"]),
                             "other_outside": int(r["other_outside"])})
                continue
            r = audit_one(token, imo, ids[imo], a.start, a.end)
            w.writerow({c: r.get(c, "") for c in cols})
            fh.flush()
            rows.append(r)
            if k % 25 == 0:
                pv.log(f"  {k}/{len(sample)}")
            time.sleep(pv.PAUSE)
    s = summary(rows)
    emit(out, "identity_audit", s)
    full_year_shares(pd.DataFrame(rows)).to_csv(out / "identity_audit_years.csv", index=False)
    pv.log(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
