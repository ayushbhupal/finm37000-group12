"""Position sizing and risk limits (issue #6).

Turns +1/-1 momentum signals into whole-contract target positions.

Method (``sizing.method: volatility_target``)
---------------------------------------------
1. Inverse-volatility weights: each product gets ``signal / vol`` so that every
   position contributes about the same risk on its own.
2. Portfolio volatility targeting: the four index futures are highly correlated
   (0.8-0.95), so summing equal per-asset risk would overshoot the target. The
   weights are scaled by ``target_annual_vol / estimated portfolio vol``, where the
   portfolio vol uses the trailing covariance matrix of daily returns.
3. Risk limits: gross leverage (sum of |notional| / capital) is capped at
   ``max_gross_leverage``, and optionally any single weight at
   ``max_weight_per_asset``.
4. Contracts: ``weight * capital / (settle * multiplier)``, rounded to the
   nearest whole contract. The leverage caps apply before this rounding, so the
   rounded book can exceed ``max_gross_leverage`` slightly (by about one contract).

All estimates on date ``t`` use data up to ``t`` only, matching the timing
convention in ``src/signals.py``. There is no stop-loss: a trend rule already
exits losing positions when the trailing return changes sign, and a separate
drawdown stop would add parameters for issue #10 to tune.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import get_spec
from .signals import TRADING_DAYS, realized_vol, returns_panel

DEFAULT_VOL_LOOKBACK = 60


def settle_panel(prices: dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """Raw settlement of the contract held *after* each date's settle, forward-filled over holidays.

    On a roll day the position moves to the next contract at that day's settle, so
    it is priced at ``next_settle``; on other days it is the held contract's ``settle``.
    """
    panel = {}
    for root, df in prices.items():
        price = df["settle"]
        if {"roll", "next_settle"} <= set(df.columns):
            on_roll = df["roll"].fillna(False).astype(bool) & df["next_settle"].notna()
            price = df["next_settle"].where(on_roll, price)
        panel[root] = price
    return pd.DataFrame(panel).reindex(index).ffill()


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Annualized volatility of a weight vector given an annualized covariance matrix."""
    w = weights.to_numpy(dtype=float)
    return float(np.sqrt(max(w @ cov.to_numpy(dtype=float) @ w, 0.0)))


def target_weights(signals: pd.DataFrame, rets: pd.DataFrame, sizing: dict,
                   vol_lookback: int = DEFAULT_VOL_LOOKBACK) -> pd.DataFrame:
    """Notional weights (fraction of capital, signed) per date and product.

    Products with no signal or no volatility estimate yet get weight 0.
    """
    method = sizing.get("method", "volatility_target")
    if method != "volatility_target":
        raise ValueError(f"Sizing method {method!r} is not implemented; use 'volatility_target'.")
    target = float(sizing["target_annual_vol"])
    max_gross = float(sizing.get("max_gross_leverage", np.inf))
    max_single = float(sizing.get("max_weight_per_asset", np.inf))

    signals = signals.reindex(index=rets.index, columns=rets.columns)
    vol = realized_vol(rets, vol_lookback)
    raw = (signals / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rets = rets.fillna(0.0)

    weights = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
    window = rets.rolling(vol_lookback, min_periods=vol_lookback)
    covs = window.cov() * TRADING_DAYS     # MultiIndex (date, product) x product
    for date in raw.index:
        w = raw.loc[date]
        if not w.any():
            continue
        active = w[w != 0].index
        cov = covs.loc[date].loc[active, active]
        if cov.isna().to_numpy().any():
            continue
        pvol = portfolio_vol(w[active], cov)
        if pvol <= 0:
            continue
        scaled = w[active] * (target / pvol)
        scaled = scaled.clip(-max_single, max_single)
        gross = scaled.abs().sum()
        if gross > max_gross:
            scaled *= max_gross / gross
        weights.loc[date, active] = scaled
    return weights


def weights_to_contracts(weights: pd.DataFrame, settles: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Round notional weights to whole contracts using each product's multiplier."""
    multipliers = pd.Series({root: get_spec(root).multiplier for root in weights.columns})
    notional_per_contract = settles[weights.columns] * multipliers
    contracts = (weights * capital / notional_per_contract).round()
    return contracts.fillna(0).astype("int64")


def size_positions(signals: pd.DataFrame, prices: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    """Whole-contract target positions per date and product.

    ``params`` is the whole config or just its ``sizing`` block (``capital``,
    ``target_annual_vol``, ``max_gross_leverage``, optional
    ``max_weight_per_asset``). The volatility window is
    ``signal.vol_lookback_days`` when the whole config is passed, else
    ``vol_lookback_days`` in the block, else 60. The position on date ``t`` is
    held from the settle of ``t`` to the settle of ``t+1``.
    """
    sizing = params.get("sizing", params)
    lookback = int(params.get("signal", {}).get("vol_lookback_days",
                                                sizing.get("vol_lookback_days", DEFAULT_VOL_LOOKBACK)))
    params = sizing
    rets = returns_panel(prices)
    weights = target_weights(signals, rets, params, vol_lookback=lookback)
    settles = settle_panel(prices, rets.index)
    return weights_to_contracts(weights, settles, float(params["capital"]))
