from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SFOC_G_PER_KWH = 190.0
CO2_PER_FUEL_T = 3.114
AUX_FRACTION_AT_SEA = 0.05
MIN_LOAD, MAX_LOAD = 0.05, 0.90
DEFAULT_SERVICE_SPEED_KN = 14.0

HFO_DENSITY_T_PER_M3 = 0.97
TANK_USABLE = 0.90

FUELS = {
    "hfo":      (40.2, 0.970),
    "lng":      (49.0, 0.450),
    "methanol": (19.9, 0.792),
    "ammonia":  (18.6, 0.682),
    "lh2":      (120.0, 0.0708),
}


def volumetric_ratio(fuel: str) -> float:
    lhv, rho = FUELS[fuel]
    hl, hr = FUELS["hfo"]
    return (lhv * rho) / (hl * hr)


def service_speed(service_kn) -> np.ndarray:
    s = np.asarray(service_kn, float)
    return np.where((s >= 5) & (s <= 30), s, DEFAULT_SERVICE_SPEED_KN)


def leg_co2_t(leg_nm, sea_hours, me_kw, service_kn, factor=1.0) -> np.ndarray:
    d = np.asarray(leg_nm, float)
    h = np.asarray(sea_hours, float)
    kw = np.asarray(me_kw, float)
    v = service_speed(service_kn)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(h > 0, d / h, np.nan)
        load = np.clip((speed / v) ** 3, MIN_LOAD, MAX_LOAD)
        fuel = (kw * load + kw * AUX_FRACTION_AT_SEA) * h * SFOC_G_PER_KWH / 1e6
    fuel = fuel * np.asarray(factor, float)
    out = fuel * CO2_PER_FUEL_T
    out[~(h > 0) | ~(kw > 0)] = np.nan
    return out


def daily_fuel_at_service_t(me_kw, factor=1.0) -> np.ndarray:
    kw = np.asarray(me_kw, float)
    return (kw * MAX_LOAD + kw * AUX_FRACTION_AT_SEA) * 24 * SFOC_G_PER_KWH / 1e6 \
        * np.asarray(factor, float)


def endurance_nm(capacity_m3, service_kn, me_kw=None, consumption_tpd=None,
                 factor=1.0, fuel="hfo") -> np.ndarray:
    cap = np.asarray(capacity_m3, float)
    v = service_speed(service_kn)
    daily = daily_fuel_at_service_t(me_kw if me_kw is not None else np.full(cap.shape, np.nan),
                                    factor)
    if consumption_tpd is not None:
        rec = np.asarray(consumption_tpd, float)
        daily = np.where(rec > 0, rec, daily)
    tonnes_hfo = cap * TANK_USABLE * HFO_DENSITY_T_PER_M3
    with np.errstate(divide="ignore", invalid="ignore"):
        days = np.where(daily > 0, tonnes_hfo / daily, np.nan)
    return days * 24 * v * volumetric_ratio(fuel)


def load_calibration(path: Path | None, types: pd.Series) -> np.ndarray:
    if path is None or not Path(path).exists():
        print("  no type_calibration.csv: CO2 weights are UNCALIBRATED")
        return np.ones(len(types))
    cal = pd.read_csv(path)
    if "fitted_on" in cal.columns:
        print(f"  calibration fitted on {cal['fitted_on'].iloc[0]}; applied to legs here")
    glob = cal.loc[cal["shiptype_group"] == "__global__", "factor"]
    glob = float(glob.iloc[0]) if len(glob) else 1.0
    fmap = dict(zip(cal["shiptype_group"], cal["factor"]))
    fmap.pop("__global__", None)
    f = pd.Series(types).map(fmap)
    print(f"  calibration: {int(f.notna().sum()):,}/{len(f):,} matched a type factor; "
          f"rest use global {glob:.2f}")
    return f.fillna(glob).to_numpy(float)


def attach_leg_co2(stops: pd.DataFrame, vessels: pd.DataFrame,
                   calibration: Path | None = None) -> np.ndarray:
    v = vessels.set_index("imo")
    types = v["shiptype_group"] if "shiptype_group" in v.columns \
        else pd.Series(index=v.index, dtype=object)
    fac = pd.Series(load_calibration(calibration, types), index=v.index)
    imo = stops["imo"]
    w = leg_co2_t(stops["leg_nm"].to_numpy(float), stops["sea_hours"].to_numpy(float),
                  imo.map(v["main_kw"]).to_numpy(float),
                  imo.map(v["service_kn"]).to_numpy(float),
                  imo.map(fac).fillna(1.0).to_numpy(float))
    return np.nan_to_num(w, nan=0.0)
