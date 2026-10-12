"""Load strategy and backtest parameters from a YAML file."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Read a YAML config into a dict.

    Relative paths are resolved against the repository root so scripts and
    notebooks behave the same regardless of the working directory.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
