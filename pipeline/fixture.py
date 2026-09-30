from __future__ import annotations

import csv
import datetime as dt
import gzip
import math
import random
from pathlib import Path

PORTS = {
    "chn-shanghai": ("SHANGHAI", "CHN", 31.357, 121.600),
    "kor-busan": ("BUSAN", "KOR", 35.075, 128.830),
    "sgp-singapore": ("SINGAPORE", "SGP", 1.264, 103.820),
    "sgp-singaporeanchorage": ("SINGAPORE ANCHORAGE", "SGP", 1.280, 103.950),
    "egy-suezcanal": ("SUEZ CANAL", "EGY", 30.000, 32.550),
    "nld-rotterdam": ("ROTTERDAM", "NLD", 51.900, 4.400),
    "nld-rotterdammaasvlakte": ("ROTTERDAM MAASVLAKTE", "NLD", 51.950, 4.050),
    "bel-antwerp": ("ANTWERP", "BEL", 51.270, 4.330),
    "usa-newyork": ("NEW YORK", "USA", 40.680, -74.150),
    "usa-losangeles": ("LOS ANGELES", "USA", 33.740, -118.260),
    "are-fujairah": ("FUJAIRAH", "ARE", 25.170, 56.360),
    "ind-mundra": ("MUNDRA", "IND", 22.740, 69.700),
    "bra-santos": ("SANTOS", "BRA", -23.980, -46.300),
    "aus-portheadland": ("PORT HEADLAND", "AUS", -20.310, 118.580),
    "esp-valencia": ("VALENCIA", "ESP", 39.440, -0.320),
    "zaf-durban": ("DURBAN", "ZAF", -29.870, 31.030),
    "mys-portklang": ("PORT KLANG", "MYS", 3.000, 101.390),
    "jpn-tokyo": ("TOKYO", "JPN", 35.620, 139.780),
    "deu-kiel": ("KIEL", "DEU", 54.330, 10.150),
    "deu-brunsbuttel": ("BRUNSBUTTEL", "DEU", 53.890, 9.140),
}
CANAL = ("deu-kiel", "deu-brunsbuttel")
PILOT_PORTS = {"swe-goteborg": ("GOTEBORG", "SWE", 57.690, 11.850)}
ALL_PORTS = PORTS | PILOT_PORTS
PILOT_SHIPS = {"9000011": "Pilot Asia-Europe", "9000012": "Pilot North Atlantic",
               "9000013": "Pilot never found"}

FIELDS = ["event_id", "imo", "start", "end", "duration_hrs", "confidence", "lat",
          "lon", "vessel_id", "ssvid", "vessel_name", "vessel_flag", "vessel_type",
          "start_anchorage_id", "start_anchorage_name", "start_anchorage_flag",
          "start_at_dock", "start_top_destination", "end_anchorage_id",
          "end_anchorage_name", "end_anchorage_flag", "end_at_dock",
          "end_top_destination"]

LINER_AE = ["chn-shanghai", "kor-busan", "sgp-singapore", "egy-suezcanal",
            "nld-rotterdam", "bel-antwerp", "nld-rotterdammaasvlakte",
            "egy-suezcanal", "sgp-singaporeanchorage"]
LINER_TP = ["chn-shanghai", "kor-busan", "usa-losangeles", "jpn-tokyo"]

SPEED_KN = 14.0
DWELL_H = 30.0


def gc_nm(a, b):
    _, _, la1, lo1 = ALL_PORTS[a]
    _, _, la2, lo2 = ALL_PORTS[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = math.radians(la2 - la1), math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 3440.065 * math.asin(math.sqrt(min(1.0, h)))


class _Writer:
    def __init__(self, fh, prefix):
        self.w = csv.DictWriter(fh, fieldnames=FIELDS)
        self.w.writeheader()
        self.n = 0
        self.prefix = prefix

    def visit(self, imo, anch, start, end, conf="4", end_anch=None):
        self.n += 1
        name, iso, lat, lon = ALL_PORTS[anch]
        ea = end_anch or anch
        self.w.writerow({k: "" for k in FIELDS} | {
            "event_id": f"{self.prefix}{self.n:06d}", "imo": imo, "start": start.isoformat(),
            "end": end.isoformat(), "confidence": conf, "lat": lat, "lon": lon,
            "vessel_flag": "PAN", "start_anchorage_id": anch,
            "start_anchorage_name": name, "start_anchorage_flag": iso,
            "start_at_dock": "False", "end_anchorage_id": ea,
            "end_anchorage_name": ALL_PORTS[ea][0], "end_anchorage_flag": ALL_PORTS[ea][1],
            "end_at_dock": "False"})


def _sail(w, imo, route, t, until, loop=True, gap_after=None):
    i, prev = 0, None
    while t < until:
        anch = route[i % len(route)]
        if prev is not None:
            t += dt.timedelta(hours=gc_nm(prev, anch) / SPEED_KN + 2)
        if gap_after is not None and t >= gap_after[0]:
            t += gap_after[1]
            gap_after = None
        e = t + dt.timedelta(hours=DWELL_H)
        w.visit(imo, anch, t, e)
        t, prev = e, anch
        i += 1
        if not loop and i >= len(route):
            break
    return t, prev


def build_fixture(d: Path, n_random: int = 30, seed: int = 7) -> dict:
    random.seed(seed)
    U = dt.timezone.utc
    T0 = dt.datetime(2019, 1, 1, tzinfo=U)
    expected = {}
    with gzip.open(d / "port_visits_0000.csv.gz", "wt", newline="") as fh:
        w = _Writer(fh, "a")
        t, _ = _sail(w, "9000001", LINER_AE, T0, dt.datetime(2022, 1, 1, tzinfo=U))
        _sail(w, "9000001", LINER_TP, t + dt.timedelta(days=20),
              dt.datetime(2024, 1, 1, tzinfo=U))
        t, _ = _sail(w, "9000002", ["aus-portheadland", "chn-shanghai"], T0,
                     dt.datetime(2022, 1, 1, tzinfo=U))
        _sail(w, "9000002", ["bra-santos", "chn-shanghai"], t + dt.timedelta(days=25),
              dt.datetime(2024, 1, 1, tzinfo=U))
        _sail(w, "9000003", ["are-fujairah", "sgp-singapore", "ind-mundra"], T0,
              dt.datetime(2024, 1, 1, tzinfo=U),
              gap_after=(dt.datetime(2020, 3, 1, tzinfo=U), dt.timedelta(days=150)))
        t = dt.datetime(2021, 5, 1, tzinfo=U)
        w.visit("9000004", "nld-rotterdam", t, t + dt.timedelta(hours=20))
        w.visit("9000004", "chn-shanghai", t + dt.timedelta(days=2),
                t + dt.timedelta(days=3))
        _sail(w, "9000004", ["chn-shanghai", "sgp-singapore"], t + dt.timedelta(days=12),
              dt.datetime(2022, 6, 1, tzinfo=U))
        _sail(w, "9000006", ["deu-kiel", "deu-brunsbuttel", "nld-rotterdam"], T0,
              dt.datetime(2020, 1, 1, tzinfo=U))
        expected["visits_written_shard0"] = w.n

    with gzip.open(d / "port_visits_0001.csv.gz", "wt", newline="") as fh:
        w = _Writer(fh, "b")
        t = dt.datetime(2020, 2, 1, tzinfo=U)
        w.visit("9000005", "esp-valencia", t, t + dt.timedelta(hours=10))
        w.visit("9000005", "bel-antwerp", t + dt.timedelta(days=4),
                t + dt.timedelta(days=5), conf="2")
        w.visit("9000005", "zaf-durban", t + dt.timedelta(days=20),
                t + dt.timedelta(days=60))
        w.visit("9000005", "nld-rotterdam", t + dt.timedelta(days=70),
                t + dt.timedelta(days=71))
        fh.write(",".join(FIELDS) + "\n")
        w.w.writerow({k: "" for k in FIELDS} | {
            "event_id": f"b{w.n:06d}", "imo": "9000005",
            "start": (t + dt.timedelta(days=70)).isoformat(),
            "end": (t + dt.timedelta(days=71)).isoformat(), "confidence": "4",
            "lat": PORTS["nld-rotterdam"][2], "lon": PORTS["nld-rotterdam"][3],
            "start_anchorage_id": "nld-rotterdam", "start_anchorage_name": "ROTTERDAM",
            "start_anchorage_flag": "NLD", "end_anchorage_id": "nld-rotterdam"})
        ids = list(PORTS)
        for v in range(n_random):
            imo = f"91{v:05d}"
            t = T0 + dt.timedelta(days=random.uniform(0, 60))
            prev = random.choice(ids)
            end_all = dt.datetime(2024, 1, 1, tzinfo=U)
            while t < end_all:
                nxt = random.choice([p for p in ids if p != prev])
                t += dt.timedelta(hours=gc_nm(prev, nxt) / SPEED_KN + random.uniform(1, 30))
                e = t + dt.timedelta(hours=random.uniform(8, 60))
                w.visit(imo, nxt, t, e)
                t, prev = e, nxt
    expected.update({"dropped_low_confidence": 1, "dropped_duration": 1,
                     "dropped_duplicate": 1})

    groups = {"9000001": "Container Ship (Fully Cellular)", "9000002": "Bulk Carrier",
              "9000003": "Crude Oil Tanker", "9000004": "Container Ship (Fully Cellular)",
              "9000005": "General Cargo Ship", "9000006": "General Cargo Ship"}
    with open(d / "ship_info.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lrnoimo_ship_no", "shiptype_level5", "shiptype_group", "ship_status",
                    "deadweight", "gross_tonnage", "teu", "year_of_build", "speedservice",
                    "total_kilowattsof_main_engines", "bunkers_descriptive_narrative",
                    "last_update_date"])
        kinds = list(groups.values())
        for imo in list(groups) + [f"91{v:05d}" for v in range(n_random)]:
            lvl = groups.get(imo, kinds[int(imo) % len(kinds)])
            cap = 1500 + 1500 * (int(imo) % 5)
            cons = "consumption: 30.50 tonnes per day" if int(imo) % 3 == 0 else ""
            w.writerow([imo, lvl, lvl.split(" ")[0], "In Service/Commission", 80000, 50000,
                        0, 2012, 14.0, 20000,
                        f"Fuel: distillate fuel: 140 cu m, residual fuel: {cap:,} cu m {cons}",
                        "2021-08-15T04:00:00Z"])

    with open(d / "type_calibration.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["shiptype_group", "n_fit", "fit_median_ratio", "factor", "fitted_on"])
        for t, f in (("Container", 1.3), ("Bulk", 1.1), ("Crude", 0.9)):
            w.writerow([t, 100, round(1 / f, 3), f, "voyages_routed.csv.gz"])
        w.writerow(["__global__", 1000, 1.0, 1.0, "voyages_routed.csv.gz"])

    with open(d / "route_cache.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dep_port", "arr_port", "routed_nm", "status"])
        for a in ALL_PORTS:
            for b in ALL_PORTS:
                r = 500.0 if {a, b} == set(CANAL) else round(1.2 * gc_nm(a, b), 3)
                w.writerow([a, b, r, "ok"])
    return expected


def build_pilot_fixture(d: Path) -> None:
    U = dt.timezone.utc
    t0, t1 = dt.datetime(2023, 1, 1, tzinfo=U), dt.datetime(2024, 1, 1, tzinfo=U)
    with gzip.open(d / "port_visits_0000.csv.gz", "wt", newline="") as fh:
        w = _Writer(fh, "p")
        _sail(w, "9000011", LINER_AE, t0 + dt.timedelta(days=3), t1)
        _sail(w, "9000012", ["nld-rotterdam", "swe-goteborg", "usa-newyork", "bel-antwerp"],
              t0 + dt.timedelta(days=9), t1)
    with open(d / "vessel_ids.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["imo", "vessel_id", "ssvid", "shipname", "flag", "n_identities",
                    "reported_imo"])
        for imo, name in PILOT_SHIPS.items():
            found = imo != "9000013"
            w.writerow([imo, f"v{imo}" if found else "", "", name.upper() if found else "",
                        "DNK", int(found), imo if found else ""])
    with open(d / "ships.csv", "w", newline="") as fh:
        fh.write("# fixture ship list, same layout as reference/methanol_ships_public.csv\n")
        w = csv.writer(fh)
        w.writerow(["imo", "name", "class", "source"])
        for imo, name in PILOT_SHIPS.items():
            w.writerow([imo, name, "fixture, 1,000 TEU", "selftest"])
