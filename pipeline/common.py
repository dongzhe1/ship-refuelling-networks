from __future__ import annotations

import math
import os
import re

import numpy as np
import pandas as pd

DEFAULT_JOBS = int(os.environ.get("JOBS") or os.cpu_count() or 4)

MIN_CONFIDENCE = 3
MAX_DURATION_HRS = 720

MAX_SEA_HOURS = 24 * 120
MAX_IMPLIED_KN = 30.0
SHORT_HOP_NM = 60.0
SLOW_KN = 3.0
SLOW_MIN_HOURS = 72.0

LEG_ORIGIN = "origin"
LEG_OK = "ok"
LEG_BREAK_GAP = "break_gap"
LEG_BREAK_SPEED = "break_speed"


def haversine_nm_vec(lat1, lon1, lat2, lon2):
    r = 3440.065
    lat1, lon1 = np.asarray(lat1, float), np.asarray(lon1, float)
    lat2, lon2 = np.asarray(lat2, float), np.asarray(lon2, float)
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def haversine_km_vec(lat1, lon1, lat2, lon2):
    return haversine_nm_vec(lat1, lon1, lat2, lon2) * 1.852


def to_utc(col):
    try:
        return pd.to_datetime(col, format="ISO8601", errors="coerce", utc=True)
    except (ValueError, TypeError):
        return pd.to_datetime(col, errors="coerce", utc=True)


_DIGITS = re.compile(r"(\d{7})")


def normalise_imo(values) -> pd.Series:
    s = pd.Series(values, dtype="object").astype(str).str.replace(r"\.0$", "", regex=True)
    return s.str.extract(_DIGITS, expand=False)


def leg_status(gc_nm, sea_hours):
    d = np.asarray(gc_nm, float)
    h = np.asarray(sea_hours, float)
    origin = np.isnan(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(h > 0, d / h, np.inf)
    gap = ~origin & (h > MAX_SEA_HOURS)
    fast = ~origin & ~gap & (np.isnan(d) | ((d > SHORT_HOP_NM) & (speed > MAX_IMPLIED_KN)))
    status = np.full(d.shape, LEG_OK, dtype=object)
    status[origin] = LEG_ORIGIN
    status[gap] = LEG_BREAK_GAP
    status[fast] = LEG_BREAK_SPEED
    slow = (status == LEG_OK) & (h > SLOW_MIN_HOURS) & (speed < SLOW_KN) & (d > SHORT_HOP_NM)
    return status, slow


def tagged(name: str, tag: str) -> str:
    if not tag:
        return name
    for ext in (".csv.gz", ".csv", ".json"):
        if name.endswith(ext):
            return f"{name[:-len(ext)]}_{tag}{ext}"
    return f"{name}_{tag}"


def fork_pool(n: int):
    import multiprocessing as mp
    if n <= 1 or "fork" not in mp.get_all_start_methods():
        return None
    return mp.get_context("fork").Pool(n)


def require(out_dir, *names, stage="build_stops.py, nodes.py and vessels.py"):
    from pathlib import Path
    missing = [n for n in names if not (Path(out_dir) / n).exists()]
    if missing:
        raise SystemExit(
            f"missing in {out_dir}: {', '.join(missing)}\n"
            f"  these are written by {stage}; run it first")


def load_stops(path, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, usecols=usecols, low_memory=False)
    for c in ("visit_start", "visit_end"):
        if c in df.columns:
            df[c] = to_utc(df[c])
    for c in ("lat", "lon", "sea_hours", "leg_gc_nm", "leg_nm", "implied_kn"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "slow" in df.columns:
        df["slow"] = df["slow"].map({"True": True, "False": False}).fillna(False).astype(bool)
    if "leg_status" in df.columns:
        df["leg_ok"] = (df["leg_status"] == LEG_OK).to_numpy()
    return df


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_floats(spec: str) -> list[float]:
    return [float(x) for x in str(spec).split(",") if x.strip()]


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in str(spec).split(",") if x.strip()]


def finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))
