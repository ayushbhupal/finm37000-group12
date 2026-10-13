"""Data download and continuous-contract construction.

Placeholder interface so other modules (and their tests) can import it now.
Implemented in issue #3 (download) and #4 (continuous contracts).
"""

import pandas as pd


def load_contract_prices(root: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV, volume and open interest for every listed contract of ``root``.

    Expected columns: date, symbol, expiry, open, high, low, close, settle,
    volume, open_interest.
    """
    raise NotImplementedError("Issue #3: pull equity futures data")


def build_continuous(prices: pd.DataFrame, roll_rule: dict) -> pd.DataFrame:
    """Stitch individual contracts into one continuous series using ``roll_rule``."""
    raise NotImplementedError("Issue #4: clean data and build continuous contracts")
