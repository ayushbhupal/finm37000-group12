"""Backtest engine.

Placeholder interface so other modules (and their tests) can import it now.
Implemented in issue #7.
"""

import pandas as pd


def run_backtest(positions: pd.DataFrame, prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Daily P&L, costs, rolls and equity curve for the target positions."""
    raise NotImplementedError("Issue #7: build the backtesting engine")
