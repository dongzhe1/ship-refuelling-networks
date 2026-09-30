from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://gateway.api.globalfishingwatch.org/v3"
IDENTITY_DS = "public-global-vessel-identity:latest"
PORTVISIT_DS = "public-global-port-visits-events:latest"
DEFAULT_START = "2023-01-01"
PAGE = 500
SHARD_ROWS = 500_000
MAX_RETRIES = 8
MAX_429_SLEEPS = 8
BASE_BACKOFF = 2.0
PAUSE = 0.12
USER_AGENT = "port-visit-extraction/1.0 (academic research; GFW API v3)"
SEARCH_LIMIT = 20
OVERLAP_DAYS = 7

MIN_CONFIDENCE = 3
MAX_DURATION_HRS = 24 * 30

FIELDS = [
    "imo", "event_id", "start", "end", "duration_hrs", "confidence",
    "lat", "lon",
    "vessel_id", "ssvid", "vessel_name", "vessel_flag", "vessel_type",
    "start_anchorage_id", "start_anchorage_name", "start_anchorage_flag",
    "start_at_dock", "start_top_destination",
    "end_anchorage_id", "end_anchorage_name", "end_anchorage_flag",
    "end_at_dock", "end_top_destination",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


DEFAULT_TOKEN_FILE = Path("~/.gfw_token")


def read_token(token_file: Path | None = None) -> str:
    p = Path(token_file or DEFAULT_TOKEN_FILE).expanduser()
    if p.exists():
        if p.stat().st_mode & 0o077:
            log(f"  warning: {p} is readable by others; chmod 600 {p}")
        tok = p.read_text().strip()
        if tok:
            return tok
        sys.exit(f"token file {p} is empty")
    if token_file is not None:
        sys.exit(f"token file {p} not found")
    tok = os.environ.get("GFW_TOKEN", "").strip()
    if not tok:
        sys.exit("no GFW token: put it in ~/.gfw_token (chmod 600), or pass --token-file FILE\n"
                 "  (the token comes from registering for API access at "
                 "globalfishingwatch.org/our-apis)")
    return tok


def read_imos(path: Path) -> list[str]:
    text = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if text and "imo" in [c.strip().lower() for c in text[0].split(",")]:
        rows = csv.DictReader(text)
        raw = [r["imo"] if "imo" in r else r["IMO"] for r in rows]
    else:
        raw = text
    out, bad = [], []
    for x in raw:
        m = re.search(r"(\d{7})", str(x))
        if not m:
            bad.append(x)
            continue
        d = [int(c) for c in m.group(1)]
        (out if sum(d[i] * (7 - i) for i in range(6)) % 10 == d[6] else bad).append(m.group(1))
    if bad:
        log(f"  dropped {len(bad)} entries with no valid IMO: {bad[:5]}")
    return list(dict.fromkeys(out))


_consec_429 = [0]


def api_get(token: str, path: str, params: list[tuple[str, str]]) -> dict | None:
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params, safe=":")
    req = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + token, "User-Agent": USER_AGENT})
    delay = BASE_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            _consec_429[0] = 0
            return d
        except (urllib.error.HTTPError, http.client.HTTPException,
                OSError, json.JSONDecodeError) as exc:
            code = getattr(exc, "code", None)
            if code in (401, 403) and attempt >= 2:
                raise SystemExit(f"auth/permission failure ({code}) -- check the token")
            if code == 422:
                try:
                    detail = json.loads(exc.read().decode())
                except Exception:
                    detail = {}
                raise SystemExit(f"422 from {path}: {detail.get('messages') or detail}\n  {url}")
            if code == 429:
                _consec_429[0] += 1
                if _consec_429[0] == 1:
                    try:
                        log(f"  429 body: {exc.read().decode()[:400]}")
                    except Exception:
                        pass
                if _consec_429[0] > MAX_429_SLEEPS:
                    raise SystemExit(
                        f"429 for {MAX_429_SLEEPS * 15} min straight -- a disabled token or a "
                        f"monthly cap, not a daily one. Progress is checkpointed; rerun later.")
                wait = 900
                try:
                    wait = max(1, min(int(exc.headers.get("Retry-After", "")), 3600))
                except (TypeError, ValueError, AttributeError):
                    pass
                log(f"  429 ({_consec_429[0]}/{MAX_429_SLEEPS}) -- sleeping {wait}s")
                time.sleep(wait)
                continue
            if attempt == MAX_RETRIES:
                log(f"  giving up on {path} after {MAX_RETRIES} tries: {exc}")
                return None
            log(f"  retry {attempt}/{MAX_RETRIES} ({type(exc).__name__} {code or ''}) "
                f"-- sleeping {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    return None


def _day(x):
    try:
        return dt.date.fromisoformat(str(x)[:10])
    except ValueError:
        return None


def pick_identities(entries: list[dict], imo: str) -> tuple[list[dict], bool]:
    def reports(e):
        return (any(str(si.get("imo") or "") == imo for si in e.get("selfReportedInfo") or [])
                or any(str(r.get("imo") or "") == imo for r in e.get("registryInfo") or []))
    out, ids = [], set()
    for e in entries:
        if not reports(e):
            continue
        for si in e.get("selfReportedInfo") or []:
            if si.get("id") and si["id"] not in ids and str(si.get("imo") or imo) == imo:
                ids.add(si["id"])
                out.append(si)
    if out:
        return out, False
    first = (entries[0].get("selfReportedInfo") or [{}])[0] if entries else {}
    return ([first] if first.get("id") else []), True


def overlaps(infos: list[dict], days: int = OVERLAP_DAYS) -> list[tuple[str, str]]:
    span = [(si.get("shipname", "?"), _day(si.get("transmissionDateFrom")),
             _day(si.get("transmissionDateTo"))) for si in infos]
    out = []
    for i in range(len(span)):
        for j in range(i + 1, len(span)):
            (a, a0, a1), (b, b0, b1) = span[i], span[j]
            if None in (a0, a1, b0, b1):
                continue
            if (min(a1, b1) - max(a0, b0)).days > days:
                out.append((a, b))
    return out


def resolve_ids(token: str, imos: list[str], cache: Path,
                identities: str = "first") -> dict[str, str]:
    known: dict[str, str] = {}
    seen: set[str] = set()
    if cache.exists():
        with open(cache, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen.add(row["imo"])
                if row["vessel_id"]:
                    known[row["imo"]] = row["vessel_id"]
        log(f"id cache: {len(seen):,} looked up, {len(known):,} matched")
    todo = [i for i in imos if i not in seen]
    if not todo:
        return {i: known[i] for i in imos if i in known}
    log(f"resolving {len(todo):,} IMOs -> vesselId")
    new = not cache.exists()
    with open(cache, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["imo", "vessel_id", "ssvid", "shipname", "flag",
                        "n_identities", "reported_imo"])
        for imo in todo:
            limit = str(SEARCH_LIMIT) if identities == "all" else "1"
            d = api_get(token, "vessels/search", [
                ("query", imo), ("datasets[0]", IDENTITY_DS), ("limit", limit)])
            if d is None:
                log(f"  unresolved (will retry): IMO {imo}")
                continue
            entries = d.get("entries") or []
            if identities == "all":
                used, fallback = pick_identities(entries, imo)
                vid = ";".join(si["id"] for si in used)
                w.writerow([imo, vid] + [";".join(str(si.get(k, "")) for si in used)
                                         for k in ("ssvid", "shipname", "flag")]
                           + [len(used), ";".join(str(si.get("imo") or "") for si in used)])
                fh.flush()
                if not used:
                    log(f"  IMO {imo}: not found in GFW")
                for si in used:
                    log(f"  IMO {imo} -> {si.get('shipname', '?')} ({si.get('flag', '?')}, "
                        f"ssvid {si.get('ssvid', '?')}, "
                        f"{str(si.get('transmissionDateFrom', '?'))[:10]} .. "
                        f"{str(si.get('transmissionDateTo', '?'))[:10]})")
                if fallback and used:
                    log(f"  WARNING: no identity reports IMO {imo}; first match used")
                for a, b in overlaps(used):
                    log(f"  WARNING: {a} and {b} were on air together for over "
                        f"{OVERLAP_DAYS} days; check for visits counted twice")
                if vid:
                    known[imo] = vid
                time.sleep(PAUSE)
                continue
            infos = (entries[0].get("selfReportedInfo") or [{}]) if entries else [{}]
            si = infos[0]
            vid = si.get("id") or ""
            reported = str(si.get("imo") or "")
            w.writerow([imo, vid, si.get("ssvid", ""), si.get("shipname", ""), si.get("flag", ""),
                        len(infos) if vid else 0, reported])
            fh.flush()
            if vid:
                known[imo] = vid
                log(f"  IMO {imo} -> {si.get('shipname', '?')} ({si.get('flag', '?')})"
                    + (f"; {len(infos)} AIS identities, only the first is pulled"
                       if len(infos) > 1 else ""))
                if reported and reported != imo:
                    log(f"  WARNING: IMO {imo} matched a ship reporting IMO {reported}")
            else:
                log(f"  IMO {imo}: not found in GFW")
            time.sleep(PAUSE)
    return {i: known[i] for i in imos if i in known}


def _num(x):
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


def flatten(e: dict, imo: str) -> dict:
    pv = e.get("port_visit") or {}
    v = e.get("vessel") or {}
    pos = e.get("position") or {}
    sa = pv.get("startAnchorage") or {}
    ea = pv.get("endAnchorage") or {}
    return {
        "imo": imo, "event_id": e.get("id"), "start": e.get("start"), "end": e.get("end"),
        "duration_hrs": pv.get("durationHrs"), "confidence": pv.get("confidence"),
        "lat": pos.get("lat"), "lon": pos.get("lon"),
        "vessel_id": v.get("id"), "ssvid": v.get("ssvid"), "vessel_name": v.get("name"),
        "vessel_flag": v.get("flag"), "vessel_type": v.get("type"),
        "start_anchorage_id": sa.get("id"), "start_anchorage_name": sa.get("name"),
        "start_anchorage_flag": sa.get("flag"), "start_at_dock": sa.get("atDock"),
        "start_top_destination": sa.get("topDestination"),
        "end_anchorage_id": ea.get("id"), "end_anchorage_name": ea.get("name"),
        "end_anchorage_flag": ea.get("flag"), "end_at_dock": ea.get("atDock"),
        "end_top_destination": ea.get("topDestination"),
    }


def keep(row: dict) -> tuple[bool, str]:
    c = _num(row.get("confidence"))
    if c is not None and c < MIN_CONFIDENCE:
        return False, "low_confidence"
    d = _num(row.get("duration_hrs"))
    if d is not None and d > MAX_DURATION_HRS:
        return False, "implausible_duration"
    return True, ""


class ShardWriter:
    def __init__(self, outdir: Path, index: int = 0):
        self.outdir, self.index, self.rows = outdir, index, 0
        self.fh = self.writer = None

    def _open(self):
        path = self.outdir / f"port_visits_{self.index:04d}.csv.gz"
        fresh = not path.exists() or path.stat().st_size == 0
        self.fh = gzip.open(path, "at", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDS)
        if fresh:
            self.writer.writeheader()

    def write(self, row: dict):
        if self.fh is None:
            self._open()
        self.writer.writerow(row)
        self.rows += 1
        if self.rows >= SHARD_ROWS:
            self.close()
            self.index += 1
            self.rows = 0

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = self.writer = None


def window(start: str | None, end: str | None, ckpt: dict) -> tuple[str, str]:
    old = (ckpt.get("start"), ckpt.get("end")) if ckpt else (None, None)
    start = start or old[0] or DEFAULT_START
    end = end or old[1] or dt.date.today().isoformat()
    for d in (start, end):
        dt.date.fromisoformat(d)
    if start >= end:
        sys.exit(f"empty window {start} .. {end}")
    if old != (None, None) and (start, end) != old:
        sys.exit(f"checkpoint.json was written for {old[0]} .. {old[1]}; use a new "
                 f"work_dir for a different window rather than mixing two in one")
    return start, end


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("work_dir", type=Path)
    ap.add_argument("--imos", type=Path, default=None,
                    help="IMO list (one per line) or CSV with an `imo` column; "
                         "default <work_dir>/imos.txt")
    ap.add_argument("--start", default=None,
                    help=f"YYYY-MM-DD; default {DEFAULT_START}, or the checkpoint's")
    ap.add_argument("--end", default=None,
                    help="YYYY-MM-DD, exclusive; default today, or the checkpoint's")
    ap.add_argument("--token-file", type=Path, default=None, help="default ~/.gfw_token")
    ap.add_argument("--identities", choices=["first", "all"], default=None,
                    help="AIS identities per IMO: the first match (the rule, "
                         "default) or all that report the IMO; default: the checkpoint's")
    a = ap.parse_args(argv)
    work = a.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    ckpt = work / "checkpoint.json"
    c = json.loads(ckpt.read_text()) if ckpt.exists() else {}
    start, end = window(a.start, a.end, c)
    old_mode = c.get("identities", "first") if c else None
    mode = a.identities or old_mode or "first"
    if old_mode and mode != old_mode:
        sys.exit(f"checkpoint.json was written with --identities {old_mode}; use a new "
                 f"work_dir rather than mixing two identity rules in one")
    if not c:
        ckpt.write_text(json.dumps({"done_imos": [], "shard_index": 0, "written": 0,
                                    "start": start, "end": end, "identities": mode}))
    imo_path = a.imos or work / "imos.txt"
    if not imo_path.exists():
        sys.exit(f"{imo_path} not found (one IMO per line, or a CSV with an imo column)")
    imos = read_imos(imo_path)
    token = read_token(a.token_file)
    log(f"{len(imos):,} IMOs from {imo_path.name}; window {start} .. {end}; "
        f"identities: {mode}")

    ids = resolve_ids(token, imos, work / "vessel_ids.csv", mode)
    done: set[str] = set()
    shard_index = written = 0
    if c.get("done_imos"):
        done = set(c["done_imos"])
        shard_index, written = c["shard_index"], c["written"]
        log(f"resuming: {len(done):,} vessels done, {written:,} rows written")

    writer = ShardWriter(work, shard_index)
    dropped = collections.Counter()
    pending = [i for i in imos if i in ids and i not in done]
    log(f"pulling port visits for {len(pending):,} vessels")
    try:
        for n, imo in enumerate(pending, 1):
            offset, failed, rows_v = 0, False, 0
            while True:
                vessels = [(f"vessels[{k}]", v) for k, v in enumerate(ids[imo].split(";"))]
                d = api_get(token, "events", [("datasets[0]", PORTVISIT_DS), *vessels,
                    ("start-date", start), ("end-date", end),
                    ("limit", str(PAGE)), ("offset", str(offset))])
                if d is None:
                    log(f"  incomplete: IMO {imo} -- will retry on next run")
                    failed = True
                    break
                entries = d.get("entries") or []
                for e in entries:
                    row = flatten(e, imo)
                    ok, why = keep(row)
                    if ok:
                        writer.write(row)
                        written += 1
                        rows_v += 1
                    else:
                        dropped[why] += 1
                nxt = d.get("nextOffset")
                if not entries or nxt is None:
                    break
                offset = nxt
                time.sleep(PAUSE)
            if not failed:
                done.add(imo)
                log(f"  {n}/{len(pending)} IMO {imo}: {rows_v} visits")
            ckpt.write_text(json.dumps({"done_imos": sorted(done), "shard_index": writer.index,
                                        "written": written, "start": start, "end": end,
                                        "identities": mode}))
            time.sleep(PAUSE)
    finally:
        writer.close()
    missing = [i for i in imos if i not in ids]
    log(f"finished: {written:,} visits from {len(done):,} vessels")
    if missing:
        log(f"not in GFW: {', '.join(missing)}")
    if dropped:
        log("dropped: " + ", ".join(f"{k}={v:,}" for k, v in dropped.most_common()))
    log(f"output: {work}")


if __name__ == "__main__":
    main()
