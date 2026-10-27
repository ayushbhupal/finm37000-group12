"""Out-of-sample and walk-forward testing (issue #10).

Every combination of the signal parameters in ``config.yaml`` (``robustness.grid``)
is backtested once over the full sample. That is valid for out-of-sample work
because signals, sizing and the backtest are causal, so a day's return does not
depend on later data. Choosing parameters then only looks at returns before a
cut-off, and judging them only at returns after it.

Returns are daily net P&L over the fixed ``sizing.capital`` (see ``src/analysis.py``).
A variant that needs a longer warm-up starts later and is NaN before that.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import analysis, backtest, portfolio, signals

TRADING_DAYS = signals.TRADING_DAYS


def variants(grid: dict[str, list]) -> list[dict]:
    """Every combination of the grid, as a list of ``{parameter: value}`` dicts."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def label(variant: dict) -> str:
    """Short readable name of a variant, e.g. ``lookback_days=252, skip_days=0``."""
    return ", ".join(f"{k}={v}" for k, v in variant.items())


def run_variant(prices: dict[str, pd.DataFrame], config: dict, variant: dict) -> pd.Series:
    """Daily fixed-capital returns of the strategy with ``variant`` overriding ``config['signal']``."""
    config = copy.deepcopy(config)
    config["signal"].update(variant)
    sig = signals.generate_signals(prices, config)
    positions = portfolio.size_positions(sig, prices, config)
    results = backtest.run_backtest(positions, prices, config)
    return analysis.returns_on_capital(results, float(config["sizing"]["capital"]))


def run_grid(prices: dict[str, pd.DataFrame], config: dict, grid: dict[str, list]) -> pd.DataFrame:
    """Returns of every variant, one column each, columns named by ``label``."""
    columns = {label(v): run_variant(prices, config, v) for v in variants(grid)}
    return pd.DataFrame(columns).sort_index()


def sharpe(returns: pd.Series | pd.DataFrame):
    """Annualised Sharpe ratio of daily returns, ignoring NaN (no risk-free rate, as in ``analysis``).

    A float for a Series, a Series with one value per column for a DataFrame, and NaN
    where the volatility is zero.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    std = frame.std()
    out = frame.mean() / std.where(std > 0) * np.sqrt(TRADING_DAYS)
    return float(out.iloc[0]) if isinstance(returns, pd.Series) else out


def split(returns: pd.DataFrame | pd.Series, split_date: str):
    """Returns before ``split_date`` (in sample) and from it onward (out of sample)."""
    cut = pd.Timestamp(split_date)
    return returns[returns.index < cut], returns[returns.index >= cut]


def in_vs_out(grid_returns: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """Sharpe of every variant in and out of sample, with the best in-sample variant flagged."""
    in_sample, out_sample = split(grid_returns, split_date)
    table = pd.DataFrame({"sharpe_in": sharpe(in_sample), "sharpe_out": sharpe(out_sample)})
    table["rank_in"] = table["sharpe_in"].rank(ascending=False)
    table["rank_out"] = table["sharpe_out"].rank(ascending=False)
    return table


def rank_correlation(table: pd.DataFrame) -> float:
    """Spearman correlation between in-sample and out-of-sample Sharpe across variants.

    Near 1 means in-sample performance predicts out-of-sample performance; near 0
    means picking the best in-sample variant is no better than picking at random.
    """
    valid = table[["sharpe_in", "sharpe_out"]].dropna()
    return float(stats.spearmanr(valid["sharpe_in"], valid["sharpe_out"])[0])


def walk_forward(grid_returns: pd.DataFrame, first_test_year: int, min_train_years: float,
                 last_year: int | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Each test year, use the variant with the best Sharpe over all earlier years.

    Training is expanding: everything before 1 January of the test year, and a year
    is skipped if fewer than ``min_train_years`` of history exist. Returns the
    stitched out-of-sample returns and one row per test year with the chosen
    variant and its training Sharpe.
    """
    last_year = last_year or grid_returns.index.max().year
    pieces, chosen = [], {}
    for year in range(first_test_year, last_year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        train = grid_returns[grid_returns.index < start]
        years = (train.dropna(how="all").index.max() - train.dropna(how="all").index.min()).days / 365.25
        if years < min_train_years:
            continue
        scores = sharpe(train).dropna()
        best = scores.idxmax()
        test = grid_returns.loc[start:pd.Timestamp(year=year, month=12, day=31), best]
        pieces.append(test)
        chosen[year] = {"variant": best, "train_sharpe": scores[best],
                        "test_sharpe": sharpe(test)}
    return pd.concat(pieces).rename("walk_forward"), pd.DataFrame(chosen).T


def cache_key(config: dict) -> str:
    """Fingerprint of everything that changes a variant's returns, to detect a stale cache."""
    parts = {k: config.get(k) for k in ("universe", "start_date", "end_date", "roll", "costs",
                                        "margin", "sizing", "signal", "robustness")}
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_or_run_grid(prices: dict[str, pd.DataFrame], config: dict, path: str | Path) -> pd.DataFrame:
    """``run_grid`` with the result cached at ``path`` (Parquet).

    The grid takes a few minutes. The cache is reused only if the config fingerprint
    stored beside it matches, so editing the grid, costs or signal settings reruns it.
    """
    path = Path(path)
    stamp = path.with_suffix(".key")
    key = cache_key(config)
    if path.exists() and stamp.exists() and stamp.read_text() == key:
        return pd.read_parquet(path)
    grid_returns = run_grid(prices, config, config["robustness"]["grid"])
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_returns.to_parquet(path)
    stamp.write_text(key)
    return grid_returns
