"""Tests for issue #6: momentum signals and volatility-targeted sizing.

Small synthetic continuous series only (the shape ``data.build_continuous``
returns), no data files.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.portfolio import portfolio_vol, size_positions, target_weights
from src.signals import TRADING_DAYS, generate_signals, momentum_signal, realized_vol, returns_panel

DATES = pd.bdate_range("2020-01-01", periods=400)
SIGNAL = {"signal": {"method": "time_series_momentum", "lookback_days": 20, "vol_lookback_days": 10}}


def continuous(rets, start_price=100.0, dates=DATES):
    """A fake ``build_continuous`` frame from a return path (first return is NaN)."""
    rets = pd.Series(rets, index=dates, dtype="float64")
    rets.iloc[0] = np.nan
    settle = start_price * (1 + rets.fillna(0)).cumprod()
    return pd.DataFrame({"settle": settle, "ret": rets, "adj_price": settle})


def noisy(drift, vol=0.01, seed=0, n=len(DATES)):
    return drift + vol * np.random.default_rng(seed).standard_normal(n)


# --- signals ---------------------------------------------------------------

def test_signal_follows_the_trend():
    # 20-day drift of +/-4% against ~1.3% of noise over the window: the sign is almost always right.
    prices = {"ES": continuous(noisy(+0.002, vol=0.003, seed=1)),
              "NQ": continuous(noisy(-0.002, vol=0.003, seed=2))}
    sig = generate_signals(prices, SIGNAL)
    late = sig.iloc[100:]
    assert (late["ES"] == 1).mean() > 0.9
    assert (late["NQ"] == -1).mean() > 0.9


def test_no_signal_until_enough_history():
    prices = {"ES": continuous(np.full(len(DATES), 0.001))}
    sig = generate_signals(prices, SIGNAL)["ES"]
    # 20 returns are needed; the first return is NaN, so day 20 (0-based) is the first signal.
    assert sig.iloc[:20].isna().all()
    assert sig.iloc[20] == 1


def test_signal_matches_trailing_compounded_return():
    rets = pd.DataFrame({"ES": noisy(0.0, seed=3)}, index=DATES)
    sig = momentum_signal(rets, lookback=20)
    t = 150
    trailing = (1 + rets["ES"].iloc[t - 19:t + 1]).prod() - 1
    assert sig["ES"].iloc[t] == np.sign(trailing)


def test_skip_days_lags_the_window():
    rets = pd.DataFrame({"ES": noisy(0.0, seed=4)}, index=DATES)
    plain = momentum_signal(rets, lookback=20)
    skipped = momentum_signal(rets, lookback=20, skip=5)
    pd.testing.assert_series_equal(skipped["ES"].iloc[30:], plain["ES"].shift(5).iloc[30:])


def test_no_look_ahead_in_signals_or_positions():
    """Changing prices after a date must not change anything dated on or before it."""
    base = {"ES": continuous(noisy(0.001, seed=5)), "NQ": continuous(noisy(-0.0005, seed=6))}
    cut = 250
    shocked = {root: df.copy() for root, df in base.items()}
    for df in shocked.values():
        df.iloc[cut + 1:, df.columns.get_loc("ret")] *= -3      # rewrite the future
    cfg = {**SIGNAL, "sizing": {"capital": 1e7, "target_annual_vol": 0.10, "max_gross_leverage": 3.0}}
    for fn in (lambda p: generate_signals(p, cfg),
               lambda p: size_positions(generate_signals(p, cfg), p, cfg)):
        a, b = fn(base), fn(shocked)
        pd.testing.assert_frame_equal(a.iloc[:cut + 1], b.iloc[:cut + 1])


def test_holidays_on_one_product_count_as_zero_return():
    es = continuous(noisy(0.001, seed=7))
    nq = continuous(noisy(0.001, seed=8)).drop(DATES[100])    # NQ closed that day
    rets = returns_panel({"ES": es, "NQ": nq})
    assert rets.loc[DATES[100], "NQ"] == 0.0
    assert rets["NQ"].iloc[0:1].isna().all()                    # before the first return: NaN


def test_late_starting_product_has_no_early_signal():
    late = continuous(noisy(0.002, seed=9, n=200), dates=DATES[200:])
    sig = generate_signals({"ES": continuous(noisy(0.002, seed=10)), "RTY": late}, SIGNAL)
    assert sig["RTY"].iloc[:220].isna().all()
    assert sig["RTY"].iloc[221:].notna().all()


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        generate_signals({"ES": continuous(noisy(0.0))}, {"method": "carry", "lookback_days": 20})


# --- sizing ----------------------------------------------------------------

def test_single_asset_hits_the_vol_target():
    rets = pd.DataFrame({"ES": noisy(0.0, vol=0.02, seed=11)}, index=DATES)
    signals = pd.DataFrame({"ES": 1.0}, index=DATES)
    w = target_weights(signals, rets, {"target_annual_vol": 0.10}, vol_lookback=60)
    vol = realized_vol(rets, 60)["ES"]
    t = 200
    assert w["ES"].iloc[t] == pytest.approx(0.10 / vol.iloc[t])


def test_correlated_assets_are_scaled_down_to_the_portfolio_target():
    rng = np.random.default_rng(12)
    common = rng.standard_normal(len(DATES)) * 0.01
    rets = pd.DataFrame({"ES": common + 0.002 * rng.standard_normal(len(DATES)),
                         "NQ": common + 0.002 * rng.standard_normal(len(DATES))}, index=DATES)
    signals = pd.DataFrame(1.0, index=DATES, columns=rets.columns)
    w = target_weights(signals, rets, {"target_annual_vol": 0.10}, vol_lookback=60)
    t = DATES[300]
    cov = rets.loc[:t].tail(60).cov() * TRADING_DAYS
    assert portfolio_vol(w.loc[t], cov) == pytest.approx(0.10)
    # Equal risk per asset would have meant 0.10 each; correlation forces less.
    single_vols = rets.loc[:t].tail(60).std() * np.sqrt(TRADING_DAYS)
    assert (w.loc[t] * single_vols < 0.10).all()


def test_gross_leverage_cap_binds():
    rets = pd.DataFrame({"ES": noisy(0.0, vol=0.001, seed=13)}, index=DATES)    # very low vol
    signals = pd.DataFrame({"ES": -1.0}, index=DATES)
    w = target_weights(signals, rets, {"target_annual_vol": 0.10, "max_gross_leverage": 2.0}, 60)
    assert w["ES"].iloc[100:].abs().max() == pytest.approx(2.0)
    assert (w["ES"].iloc[100:] < 0).all()                                       # sign kept


def test_contracts_are_whole_and_use_the_multiplier():
    es = continuous(noisy(0.001, vol=0.01, seed=14), start_price=4000.0)
    cfg = {**SIGNAL, "sizing": {"capital": 1e7, "target_annual_vol": 0.10, "max_gross_leverage": 3.0}}
    sig = generate_signals({"ES": es}, cfg)
    pos = size_positions(sig, {"ES": es}, cfg)
    assert pos.dtypes.eq("int64").all()
    t = DATES[300]
    w = 0.10 / realized_vol(returns_panel({"ES": es}), 10).loc[t, "ES"] * sig.loc[t, "ES"]
    expected = round(min(abs(w), 3.0) * np.sign(w) * 1e7 / (es.loc[t, "settle"] * 50))
    assert pos.loc[t, "ES"] == expected
    assert (pos["ES"].iloc[:20] == 0).all()                                    # flat while warming up


def test_repo_config_has_the_fields_sizing_needs():
    cfg = load_config()
    assert cfg["sizing"]["capital"] > 0
    assert cfg["signal"]["method"] == "time_series_momentum"
    assert cfg["signal"]["vol_lookback_days"] >= 2
