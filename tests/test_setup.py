"""Smoke tests for issue #1: the package installs, imports and loads its config."""

import importlib

import pytest

from src.config import load_config
from src.contracts import CONTRACTS, get_spec

MODULES = ["src", "src.contracts", "src.config", "src.data", "src.signals",
           "src.portfolio", "src.backtest", "src.analysis"]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name):
    importlib.import_module(name)


def test_config_loads_and_matches_contracts():
    config = load_config()
    assert config["universe"], "universe must not be empty"
    for root in config["universe"]:
        assert root in CONTRACTS


@pytest.mark.parametrize(
    "root, tick_value",
    [("ES", 12.50), ("NQ", 5.00), ("RTY", 5.00), ("YM", 5.00)],
)
def test_tick_values(root, tick_value):
    assert get_spec(root).tick_value == pytest.approx(tick_value)


def test_symbol_and_notional():
    es = get_spec("es")
    assert es.symbol(2026, 3) == "ESH6"
    assert es.symbol(2026, 6) == "ESM6"
    assert es.notional(5500.0, 2) == pytest.approx(550_000.0)
    with pytest.raises(ValueError):
        es.symbol(2026, 4)
