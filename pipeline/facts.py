from __future__ import annotations

import json
from pathlib import Path


def emit(results: Path, name: str, values: dict) -> Path:
    out = Path(results) / f"facts_{name}.json"
    clean = {}
    for k, v in values.items():
        if isinstance(v, bool):
            clean[k] = v
        elif isinstance(v, (int, float)):
            f = float(v)
            if f != f or f in (float("inf"), float("-inf")):
                raise ValueError(f"{name}.{k} is {f}, refusing to record it")
            clean[k] = int(v) if isinstance(v, int) else f
        else:
            clean[k] = str(v)
    out.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")
    print(f"  facts -> {out.name}  ({len(clean)} values)")
    return out
