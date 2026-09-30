from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture

LAYOUT = {
    "port_visits_0000.csv.gz": "gfw/port_visits_0000.csv.gz",
    "port_visits_0001.csv.gz": "gfw/port_visits_0001.csv.gz",
    "ship_info.csv": "seaweb/ship_info.csv",
    "type_calibration.csv": "type_calibration.csv",
    "route_cache.csv": "route_cache.csv",
}


def write(out: Path, ships: int = 30, seed: int = 7) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        expected = fixture.build_fixture(t, n_random=ships, seed=seed)
        for src, dst in LAYOUT.items():
            (out / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(t / src), out / dst)
    pull = out / "methanol_pull"
    pull.mkdir(exist_ok=True)
    fixture.build_pilot_fixture(pull)
    expected = {"ships_random": ships, "seed": seed, **expected}
    (out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n")
    return expected


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--ships", type=int, default=30,
                    help="vessels of random traffic beside the scripted cases (default 30)")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    out = a.dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    write(out, a.ships, a.seed)
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
