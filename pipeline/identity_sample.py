from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
N_SAMPLE, SEED, GROUP = 1000, 20260929, "container"


def candidates(vessels: pd.DataFrame, ids: pd.DataFrame, group: str) -> list[str]:
    v = set(vessels.loc[vessels["group"] == group, "imo"].astype(str))
    have = set(ids.loc[ids["vessel_id"].fillna("").astype(str) != "", "imo"].astype(str))
    return sorted(v & have)


def draw(pool: list[str], n: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("main_out", type=Path)
    ap.add_argument("gfw_dir", type=Path)
    ap.add_argument("out_csv", type=Path)
    ap.add_argument("--n", type=int, default=N_SAMPLE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--group", default=GROUP)
    a = ap.parse_args(argv)
    out = a.out_csv.expanduser().resolve()
    if (HERE / "reference").resolve() in out.parents:
        sys.exit("out_csv is inside pipeline/reference/: the list is derived from the "
                 "licensed register and must stay out of the code directory")
    gfw = a.gfw_dir.expanduser().resolve()
    if gfw in out.parents:
        sys.exit("out_csv must not be inside the GFW directory (read only)")
    vessels = pd.read_csv(a.main_out / "vessels.csv", dtype={"imo": str})
    ids = pd.read_csv(gfw / "vessel_ids.csv", dtype=str)
    pool = candidates(vessels, ids, a.group)
    ships = draw(pool, a.n, a.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# identity-repair subsample: {len(ships)} of "
                 f"{len(pool)} {a.group} ships, seed {a.seed}. Derived from the licensed "
                 f"register: keep on the cluster.\n")
        pd.DataFrame({"imo": ships, "group": a.group}).to_csv(fh, index=False)
    print(f"{len(pool):,} {a.group} ships with an identity; {len(ships)} drawn -> {out}")


if __name__ == "__main__":
    main()
