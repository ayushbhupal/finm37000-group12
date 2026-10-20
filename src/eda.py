"""Exploratory data analysis helpers (issue #5).

Small, tested functions that ``notebooks/05_eda.ipynb`` imports, so the notebook
holds the charts and the write-up rather than the logic. Everything here works on
the continuous series from ``data.build_continuous`` (``ret`` is the daily simple
return of the held contract, so rolls create no jumps).
"""

import numpy as np
import pandas as pd
from scipy import stats

from . import data

TRADING_DAYS = 252


def load_universe(config: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Raw contract prices and continuous series for each product in ``config['universe']``."""
    prices, continuous = {}, {}
    for root in config["universe"]:
        prices[root] = data.load_contract_prices(root, config["start_date"], config["end_date"])
        continuous[root] = data.build_continuous(prices[root], config["roll"])
    return prices, continuous


def return_panel(continuous: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Daily returns, one column per product, aligned on trade date (NaN before a product starts)."""
    return pd.DataFrame({root: cont["ret"] for root, cont in continuous.items()}).sort_index()


def return_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-product distribution statistics of daily returns.

    ``ann_return`` and ``ann_vol`` scale the daily mean and standard deviation by
    252 and sqrt(252). ``excess_kurtosis`` is 0 for a normal distribution, and
    ``jarque_bera_p`` is the p-value of the normality test.
    """
    rows = {}
    for root, ret in returns.items():
        ret = ret.dropna()
        rows[root] = {
            "start": ret.index.min(), "days": len(ret),
            "ann_return": ret.mean() * TRADING_DAYS, "ann_vol": ret.std() * np.sqrt(TRADING_DAYS),
            "skew": stats.skew(ret), "excess_kurtosis": stats.kurtosis(ret),
            "min": ret.min(), "max": ret.max(),
            "pct_beyond_3sd": (np.abs(ret - ret.mean()) > 3 * ret.std()).mean(),
            "jarque_bera_p": stats.jarque_bera(ret).pvalue,
        }
    return pd.DataFrame(rows).T


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised rolling standard deviation of daily returns.

    Each product is rolled over its own trading days, so a date missing from one
    product's data does not blank the next ``window`` days.
    """
    return pd.DataFrame({
        root: ret.dropna().rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)
        for root, ret in returns.items()
    }).reindex(returns.index)


def missing_days(continuous: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dates a product has no row although another product does, after its first date.

    Exchange holidays are missing everywhere and do not show up here. A date listed
    for one product only points to a gap in that product's source data.
    """
    calendar = pd.DatetimeIndex(sorted(set().union(*(cont.index for cont in continuous.values()))))
    rows = []
    for root, cont in continuous.items():
        gaps = calendar[(calendar > cont.index.min()) & ~calendar.isin(cont.index)]
        rows += [{"product": root, "date": day} for day in gaps]
    return pd.DataFrame(rows, columns=["product", "date"])


def autocorrelation(returns: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Autocorrelation of daily returns at each lag (rows) for each product (columns)."""
    return pd.DataFrame({root: [ret.dropna().autocorr(lag) for lag in lags]
                         for root, ret in returns.items()}, index=pd.Index(lags, name="lag"))


def autocorrelation_band(n: int) -> float:
    """Approximate 95% band for the autocorrelation of ``n`` independent observations."""
    return 1.96 / np.sqrt(n)


def trailing_return(ret: pd.Series, lookback: int) -> pd.Series:
    """Compounded return over the ``lookback`` days up to and including each date."""
    return (1 + ret).rolling(lookback, min_periods=lookback).apply(np.prod, raw=True) - 1


def momentum_persistence(returns: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    """Does the sign of the trailing return predict the next ``horizon`` days?

    Samples every ``horizon`` days so forward windows do not overlap. The signal
    on date t is the sign of the trailing ``lookback``-day return, and the forward
    return compounds days t+1 to t+horizon. ``strategy_mean`` is the average
    signal-times-forward return per period (long when up, short when down),
    ``t_stat`` its t-statistic, and ``hit_rate`` the share of periods with a
    positive result. ``mean_fwd_all`` is the average forward return whatever the
    signal, the baseline for always being long. This is an in-sample description,
    not a backtest.
    """
    rows = {}
    for root, ret in returns.items():
        ret = ret.dropna()
        signal = np.sign(trailing_return(ret, lookback))
        forward = (1 + ret).rolling(horizon).apply(np.prod, raw=True).shift(-horizon) - 1
        sample = pd.DataFrame({"signal": signal, "forward": forward}).dropna().iloc[::horizon]
        result = sample["signal"] * sample["forward"]
        rows[root] = {
            "periods": len(sample),
            "pct_long": (sample["signal"] > 0).mean(),
            "mean_fwd_after_up": sample.loc[sample["signal"] > 0, "forward"].mean(),
            "mean_fwd_after_down": sample.loc[sample["signal"] < 0, "forward"].mean(),
            "mean_fwd_all": sample["forward"].mean(),
            "strategy_mean": result.mean(),
            "t_stat": result.mean() / (result.std() / np.sqrt(len(result))),
            "hit_rate": (result > 0).mean(),
        }
    return pd.DataFrame(rows).T


def term_structure(prices: pd.DataFrame, continuous: pd.DataFrame) -> pd.DataFrame:
    """Front and next-contract settles for each date, with the annualised roll yield.

    The front contract is the one held in ``continuous`` and the next is the
    following expiry. ``roll_yield`` is ln(front / next) scaled to a year by the
    days between the two expiries: positive means backwardation (the front is
    dearer, so a long position earns the roll), negative means contango.
    """
    settles = data.settle_table(data.clean_settles(prices))
    expiries = settles.columns
    position = expiries.get_indexer(continuous["expiry"]) + 1
    has_next = position < len(expiries)
    next_expiry = pd.DatetimeIndex(expiries[np.where(has_next, position, 0)])
    rows = np.arange(len(continuous))
    next_settle = settles.reindex(continuous.index).to_numpy()[rows, np.where(has_next, position, 0)]
    out = pd.DataFrame({
        "front_expiry": continuous["expiry"], "next_expiry": next_expiry,
        "front": continuous["settle"], "next": np.where(has_next, next_settle, np.nan),
    }, index=continuous.index)
    out.loc[~has_next, "next_expiry"] = pd.NaT
    gap_days = (out["next_expiry"] - out["front_expiry"]).dt.days
    out["roll_yield"] = np.log(out["front"] / out["next"]) * 365 / gap_days
    return out


def roll_summary(continuous: pd.DataFrame) -> pd.Series:
    """Cost of the rolls in a continuous series.

    ``gap`` is next / front - 1 at each roll: positive (contango) means a long
    position buys the next contract dearer than it sold the front. ``per_year``
    is the mean gap times the rolls per year, a rough annual drag (or gain) of
    holding a long position through the rolls.
    """
    rolls = continuous[continuous["roll"]]
    gap = rolls["next_settle"] / rolls["settle"] - 1
    years = (continuous.index.max() - continuous.index.min()).days / 365.25
    return pd.Series({
        "rolls": len(rolls), "mean_gap": gap.mean(), "pct_contango": (gap > 0).mean(),
        "rolls_per_year": len(rolls) / years, "per_year": gap.mean() * len(rolls) / years,
    })
