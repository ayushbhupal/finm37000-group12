"""Signal generation (issue #6): time-series momentum.

The strategy chosen in issue #2 is time-series momentum (Moskowitz, Ooi and
Pedersen, 2012): each contract is long when its own trailing return is positive
and short when it is negative, independently of the others.

Timing convention (no look-ahead)
---------------------------------
The value on date ``t`` uses only settlements up to and including ``t``. It is
the signal for the position held **from the settle of t to the settle of t+1**,
so the backtest (issue #7) earns ``ret`` on ``t+1`` with the position set on
``t``. Every function here is causal: changing a price after ``t`` cannot change
anything dated ``t`` or earlier (see ``tests/test_signals.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def returns_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily same-contract returns, one column per product, on a common calendar.

    ``prices`` maps root -> output of ``data.build_continuous``. Products trade on
    slightly different calendars, so the dates are unioned; a product with no
    settlement on a date (its own holiday) gets a return of 0 that day, which is
    what holding the position through the holiday earns. Dates before a
    product's first return stay NaN, so it does not count as "history".
    """
    rets = pd.DataFrame({root: df["ret"] for root, df in prices.items()}).sort_index()
    started = rets.notna().cummax()
    return rets.fillna(0.0).where(started)


def momentum_signal(rets: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """Sign of the trailing ``lookback``-day compounded return, ending ``skip`` days ago.

    Returns +1 (long), -1 (short) or 0 (flat, an exactly zero trailing return).
    The first ``lookback + skip`` dates of each product are NaN: not enough
    history to form a signal yet.
    """
    if lookback < 1 or skip < 0:
        raise ValueError("lookback must be >= 1 and skip >= 0")
    log_growth = np.log1p(rets)
    trailing = log_growth.rolling(lookback, min_periods=lookback).sum().shift(skip)
    return np.sign(trailing)


def realized_vol(rets: pd.DataFrame, window: int) -> pd.DataFrame:
    """Trailing annualized volatility of daily returns (NaN until ``window`` days exist)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return rets.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def generate_signals(prices: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    """Signal per date and product (+1 / -1 / 0, NaN while warming up).

    ``params`` is either the whole config or just its ``signal`` block, with
    ``method`` (only ``time_series_momentum`` is implemented), ``lookback_days``
    and optionally ``skip_days`` (default 0; 21 skips the most recent month, a
    common variant that avoids short-term reversal).
    """
    sig_cfg = params.get("signal", params)
    method = sig_cfg.get("method", "time_series_momentum")
    if method != "time_series_momentum":
        raise ValueError(f"Signal method {method!r} is not implemented; use 'time_series_momentum'.")
    rets = returns_panel(prices)
    return momentum_signal(rets, int(sig_cfg["lookback_days"]), int(sig_cfg.get("skip_days", 0)))
