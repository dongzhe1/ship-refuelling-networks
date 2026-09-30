# ship-refuelling-networks

Code to measure, from AIS port calls and a ship register, how much ship traffic
a network of refuelling ports can serve when the fuel has a limited range, and
how this varies with the choice of ports, tank size and year. It reproduces the
result files in `results/`.

---

## System Requirements

### Software Dependencies

All Python package dependencies are listed in `requirements.txt`. Key libraries:

| Package | Version |
|---|---|
| Python | ≥ 3.11 |
| pandas | 3.0.3 |
| numpy | 2.3.5 |
| searoute | 1.6.0 |

> **Note:** Python 3.11 or above is required. The result files in `results/`
> were produced with Python 3.11.5. `run.sh` needs bash (on Windows, Git Bash
> or WSL).

### Tested Operating Systems

- macOS 26.2
- Ubuntu 22.04
- Windows 11

### Hardware

No non-standard hardware is required. The demo runs on a normal desktop
computer. The full run reads 7.9 million port visits; its largest steps need up
to 160 GB of memory, and it uses all CPU cores by default (`JOBS` sets the
number of processes). No GPU is used.

---

## Installation

```bash
pip install -r requirements.txt
```

**Typical installation time:** 1–2 minutes on a standard desktop computer,
depending on network speed.

---

## Demo

> **Note on data:** the full input data are not publicly available: the ship
> register is licensed, and the port-visit events are pulled per ship from a
> licensed ship list. The demo uses **synthetically generated fake data**
> (`pipeline/make_fake_inputs.py`) solely to verify that the pipeline runs
> correctly end to end. The fake ships and voyages are invented, so the results
> **cannot be used to evaluate the actual findings**.

```bash
bash run.sh fake
```

This generates the fake inputs under `fake/inputs/`, in the same formats as the
real data (port visits, ship register, calibration factors, route cache,
methanol-ship pull), runs all 35 steps of the pipeline on them with small
settings, and writes 67 result files to `fake/results/`.

Expected output: one line per step, ending with the location of the results:

```
--- make_fake_inputs.py ...
--- build_stops.py ...
--- route_distances.py ...
...
--- fueleu.py ...
results -> .../fake/results
```

**Expected run time:** under a minute on a normal desktop computer (26 seconds on 16 cores).

---

## Data Requirements

The full run reads the following inputs. `bash run.sh fake` writes an example of
each to `fake/inputs/`.

### 1. Port visits: `port_visits_*.csv.gz`
One row per port visit, as written by `pipeline/pull_port_visits.py`:
```
event_id, imo, start, end, confidence, lat, lon, start_anchorage_id,
start_anchorage_name, start_anchorage_flag, start_at_dock, end_anchorage_id
```

### 2. Ship register: `ship_info.csv`
A Sea-web export (S&P Global; licensed, not included):
```
lrnoimo_ship_no, shiptype_level5, shiptype_group, ship_status, deadweight,
gross_tonnage, teu, year_of_build, speedservice, total_kilowattsof_main_engines,
bunkers_descriptive_narrative, last_update_date
```

### 3. Calibration factors: `type_calibration.csv` (included in `data/`)
```
shiptype_group, n_fit, fit_median_ratio, factor, fitted_on
```

### 4. Route cache (optional): `route_distances.csv`
```
dep_port, arr_port, routed_nm, status
```

**Key Fields**:
* **imo / lrnoimo_ship_no**: IMO ship number
* **start, end**: start and end of the visit (UTC)
* **confidence**: GFW visit confidence; visits below 3 are dropped
* **start_anchorage_id, end_anchorage_id**: GFW anchorage identifiers
* **start_anchorage_flag**: country of the anchorage (ISO 3166-1 alpha-3)
* **shiptype_level5, shiptype_group**: ship type, mapped to container, bulk, tanker, gas, general cargo, ro-ro
* **teu, speedservice, total_kilowattsof_main_engines**: container capacity, service speed (knots), main-engine power (kW)
* **bunkers_descriptive_narrative**: fuel tank text, parsed for tank volume (m³) and daily consumption (t/day)
* **factor**: multiplier on modelled fuel use per ship type; the `__global__` row applies to types not listed
* **routed_nm**: sea distance between two anchorages (nautical miles)

### Sources and licences

| Input | Source | Included |
|---|---|---|
| Port-visit events | Global Fishing Watch API, CC BY-NC 4.0 | no; `pipeline/pull_port_visits.py` |
| Ship register | Sea-web (S&P Global), licensed | no |
| External port sets, documented methanol bunkering deliveries, methanol ships with public IMO numbers, transit anchorages, regions | public sources | `pipeline/reference/` |
| Result files | this code | `results/` |
| Fuel-use calibration factors per ship type, fitted on public EU MRV reports | derived aggregate | `data/type_calibration.csv` |
Quantities derived from the ship register enter `results/` only aggregated.

---

## Data Preprocessing

**Port visits.** Events are pulled vessel by vessel from the Global Fishing
Watch API for 2018-01-01 to 2026-08-01, because a date-window query returns
events that overlap the window rather than start in it. Visits with confidence
below 3 or lasting more than 30 days are dropped, as are exact duplicates. The
result files use 29,209 IMO numbers from the Sea-web register snapshot of
15 August 2021 (28,501 found by GFW, 27,150 with valid visits). GFW updates its
data continuously, so a new pull differs slightly from the one used here
(August–September 2026).

**Legs and ports.** Consecutive visits of a ship form legs, measured on routed
sea distance (`searoute`), or on the great circle where a route cannot be
sailed in the time available. A leg breaks the chain of calls when the ship is
silent for more than 120 days or would need more than 30 knots. Anchorages
within 30 km in the same country are merged into refuelling nodes.

**Ships.** Ship type, container capacity, engine power and service speed come
from the register; tank volume and daily consumption are parsed from its
bunker text. Fuel use per leg follows the cube of speed over service speed,
scaled by the calibration factors.

---

## Usage

### 1. Pull the port visits

Get a free non-commercial token at <https://globalfishingwatch.org/our-apis/>
and save it in `~/.gfw_token` (`chmod 600`). Write the IMO numbers, one per
line, to `imos.txt`, then:

```bash
python pipeline/pull_port_visits.py /data/gfw --imos imos.txt --start 2018-01-01 --end 2026-08-01
```

A pull of 29,000 ships takes about a day. The methanol ships (optional):

```bash
python pipeline/pull_port_visits.py /data/methanol --imos pipeline/reference/methanol_ships_public.csv \
       --start 2023-01-01 --end 2026-09-01 --identities all
python pipeline/pull_port_visits.py /data/methanol_ext --imos pipeline/reference/methanol_ships_public_ext.csv \
       --start 2023-01-01 --end 2026-09-01 --identities all
```

### 2. Run

```bash
export GFW_DIR=/data/gfw SEAWEB=/data/ship_info.csv OUT_DIR=/data/out
export METHANOL_DIR=/data/methanol METHANOL_EXT_DIR=/data/methanol_ext     # optional
export CALIBRATION=$PWD/data/type_calibration.csv
export ROUTE_CACHE=/data/route_distances.csv                               # optional
bash run.sh full
```

The result files are written to `$OUT_DIR/bundle/`, laid out as `results/`;
`results/MANIFEST.txt` gives the row count and a hash of every file.

### 3. Result sets that need their own pulls

```bash
python pipeline/identity_sample.py $OUT_DIR $GFW_DIR $OUT_DIR/identity_rerun/ships.csv
python pipeline/pull_port_visits.py /data/idsample --imos $OUT_DIR/identity_rerun/ships.csv \
       --start 2018-01-01 --end 2026-08-01 --identities all
IDSAMPLE_DIR=/data/idsample bash run.sh identity

python pipeline/audit_identities.py $GFW_DIR $OUT_DIR/identity_audit --n 300
bash run.sh collect
```

---

## Repository Structure

```
run.sh                 entry point: fake | full | identity | collect | restore
requirements.txt
pipeline/              the pipeline, one script per step
pipeline/reference/    public reference inputs
data/                  fuel-use calibration factors per ship type
results/               the result files; MANIFEST.txt
```

---

## License

Port-visit data: Copyright 2026, Global Fishing Watch, Inc.,
https://globalfishingwatch.org/our-apis/, licensed CC BY-NC 4.0; use for
non-commercial purposes only. The code licence will be set when the deposit is
made on acceptance.
