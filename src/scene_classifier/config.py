from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate an experiment YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    required = {"run_name", "seed", "data", "model", "training"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")
    return config
