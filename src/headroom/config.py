"""Configuration loading, paths, seeding and logging."""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path
    raw: Path
    processed: Path
    tables: Path
    figures: Path
    logs: Path

    def ensure(self) -> Paths:
        for p in (self.raw, self.processed, self.tables, self.figures, self.logs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None, quick: bool | None = None) -> dict[str, Any]:
    """Read config.yaml. Quick mode (env HEADROOM_QUICK=1) shrinks replicate counts."""
    path = path or ROOT / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if quick is None:
        quick = os.environ.get("HEADROOM_QUICK", "0") == "1"
    if quick:
        cfg = _deep_update(cfg, cfg.get("quick", {}))
    cfg["_quick"] = bool(quick)
    return cfg


def get_paths(cfg: dict[str, Any]) -> Paths:
    return Paths(
        root=ROOT,
        raw=ROOT / cfg["data"]["raw_dir"],
        processed=ROOT / cfg["data"]["processed_dir"],
        tables=ROOT / "results" / "tables",
        figures=ROOT / "results" / "figures",
        logs=ROOT / "results" / "logs",
    ).ensure()


def rng(cfg: dict[str, Any], offset: int = 0) -> np.random.Generator:
    """Independent, reproducible random stream per analysis step (seed + offset)."""
    return np.random.default_rng(cfg["seed"] + offset)


def get_logger(name: str, paths: Paths | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if paths is not None:
        fh = logging.FileHandler(paths.logs / f"{name}.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def write_environment(paths: Paths, cfg: dict[str, Any]) -> None:
    """Record software versions and config used for the run (provenance)."""
    import matplotlib
    import pandas
    import scipy
    import sklearn
    import statsmodels

    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
        "quick_mode": cfg["_quick"],
        "seed": cfg["seed"],
    }
    (paths.logs / "environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
