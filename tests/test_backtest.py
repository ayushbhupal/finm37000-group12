"""Tests for issue #7: the backtest engine.

Small hand-computed cases on fake continuous series (the shape
``data.build_continuous`` returns), no data files. ES: multiplier 50, tick value
12.50. NQ: multiplier 20, tick value 5.00.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest import RESULT_COLUMNS, run_backtest
from src.portfolio import settle_panel, size_positions
from src.signals import generate_signals

CFG = {
    "costs": {"commission_per_contract": 2.50, "slippage_ticks": 1},
    "margin": {"pct_of_notional": 0.08},
    "sizing": {"capital": 1_000_000},
}
ES_COST = 2.50 + 12.50      # commission + one tick of slippage, per contract traded
NQ_COST = 2.50 + 5.00


def dates(n):
    return pd.bdate_range("2024-01-02", periods=n)


def continuous(settle, pnl_points, index, roll=None, next_settle=None):
    """A fake ``build_continuous`` frame; ``pnl_points`` is given, not derived."""
    n = len(index)
    return pd.DataFrame({
        "settle": settle,
        "pnl_points": pnl_points,
        "roll": roll if roll is not None else [False] * n,
        "next_settle": next_settle if next_settle is not None else [np.nan] * n,
    }, index=index)


def book(index, **cols):
    return pd.DataFrame(cols, index=index, dtype="int64")


# --- the known case --------------------------------------------------------

def test_buy_and_hold_one_product_matches_hand_calculation():
    idx = dates(4)
    es = continuous([4000.0, 4010.0, 4005.0, 4020.0], [np.nan, 10.0, -5.0, 15.0], idx)
    res = run_backtest(book(idx, ES=[2, 2, 2, 2]), {"ES": es}, CFG)

    # P&L is 2 contracts * points * $50, earned from the day after the position is set.
    assert res["gross_pnl"].tolist() == [0.0, 1000.0, -500.0, 1500.0]
    assert res["contracts_traded"].tolist() == [2, 0, 0, 0]
    assert res["costs"].tolist() == [2 * ES_COST, 0.0, 0.0, 0.0]
    assert res["equity"].tolist() == pytest.approx(
        [1_000_000 - 30, 1_000_970, 1_000_470, 1_001_970])
    assert res["margin"].iloc[0] == pytest.approx(0.08 * 2 * 4000.0 * 50)
    assert res["gross_notional"].iloc[-1] == pytest.approx(2 * 4020.0 * 50)
    assert list(res.columns) == RESULT_COLUMNS + ["pos_ES"]


def test_short_position_profits_when_the_price_falls():
    idx = dates(3)
    es = continuous([4000.0, 3990.0, 3970.0], [np.nan, -10.0, -20.0], idx)
    res = run_backtest(book(idx, ES=[-1, -1, -1]), {"ES": es}, CFG)
    assert res["gross_pnl"].tolist() == [0.0, 500.0, 1000.0]
    assert res["gross_notional"].iloc[0] == pytest.approx(4000.0 * 50)   # notional is absolute


# --- timing ----------------------------------------------------------------

def test_a_position_earns_nothing_on_the_day_it_is_set():
    idx = dates(5)
    es = continuous([4000.0] * 5, [np.nan, 10.0, 10.0, 10.0, 10.0], idx)
    res = run_backtest(book(idx, ES=[0, 0, 1, 1, 1]), {"ES": es}, CFG)
    assert res.index[0] == idx[2]                  # output starts when the first position is held
    assert res["gross_pnl"].iloc[0] == 0.0         # set at idx[2], earns from idx[3]
    assert res["gross_pnl"].iloc[1] == 500.0


def test_resizing_pays_on_the_change_and_earns_on_the_new_size_next_day():
    idx = dates(4)
    es = continuous([4000.0] * 4, [np.nan, 10.0, 10.0, 10.0], idx)
    res = run_backtest(book(idx, ES=[1, 3, 3, 3]), {"ES": es}, CFG)
    assert res["gross_pnl"].tolist() == [0.0, 500.0, 1500.0, 1500.0]      # earns on the size held
    assert res["contracts_traded"].tolist() == [1, 2, 0, 0]
    assert res["costs"].tolist() == [ES_COST, 2 * ES_COST, 0.0, 0.0]


def test_a_flip_trades_the_full_swing():
    idx = dates(3)
    es = continuous([4000.0] * 3, [np.nan, 0.0, 0.0], idx)
    res = run_backtest(book(idx, ES=[-1, 2, 2]), {"ES": es}, CFG)
    assert res["contracts_traded"].tolist() == [1, 3, 0]


def test_future_pnl_does_not_change_the_past():
    idx = dates(40)
    rng = np.random.default_rng(0)
    points = rng.normal(0, 10, 40)
    points[0] = np.nan
    pos = book(idx, ES=rng.integers(-3, 4, 40))
    base = run_backtest(pos, {"ES": continuous([4000.0] * 40, points, idx)}, CFG)
    shocked_points = points.copy()
    shocked_points[26:] *= -5
    shocked = run_backtest(pos, {"ES": continuous([4000.0] * 40, shocked_points, idx)}, CFG)
    assert base.notna().all().all()
    pd.testing.assert_frame_equal(base.loc[:idx[25]], shocked.loc[:idx[25]])


# --- rolls -----------------------------------------------------------------

def test_a_roll_trades_both_legs_and_prices_the_new_contract_at_next_settle():
    idx = dates(5)
    # Roll on idx[2]: the old contract settles 4040, the new one 4050. The next day's
    # move is measured from 4050, so 4062 is +12 points, not +22.
    es = continuous([4000.0, 4010.0, 4040.0, 4062.0, 4055.0],
                    [np.nan, 10.0, 30.0, 12.0, -7.0], idx,
                    roll=[False, False, True, False, False],
                    next_settle=[np.nan, np.nan, 4050.0, np.nan, np.nan])
    res = run_backtest(book(idx, ES=[3, 3, 3, 3, 3]), {"ES": es}, CFG)

    assert res["gross_pnl"].tolist() == [0.0, 1500.0, 4500.0, 1800.0, -1050.0]
    assert res["contracts_traded"].tolist() == [3, 0, 6, 0, 0]              # 3 closed + 3 opened
    assert res["costs"].iloc[2] == pytest.approx(6 * ES_COST)
    assert res["gross_notional"].iloc[1] == pytest.approx(3 * 4010.0 * 50)  # old contract
    assert res["gross_notional"].iloc[2] == pytest.approx(3 * 4050.0 * 50)  # new contract at next_settle


def test_a_roll_day_resize_pays_for_both_legs_in_full():
    idx = dates(3)
    es = continuous([4000.0, 4010.0, 4020.0], [np.nan, 0.0, 0.0], idx,
                    roll=[False, True, False], next_settle=[np.nan, 4015.0, np.nan])
    res = run_backtest(book(idx, ES=[2, 5, 5]), {"ES": es}, CFG)
    assert res["contracts_traded"].tolist() == [2, 7, 0]                    # close 2, open 5


def test_sizing_prices_a_roll_day_at_next_settle():
    idx = dates(3)
    es = continuous([4000.0, 4040.0, 4062.0], [np.nan, 40.0, 12.0], idx,
                    roll=[False, True, False], next_settle=[np.nan, 4050.0, np.nan])
    panel = settle_panel({"ES": es}, idx)
    assert panel["ES"].tolist() == [4000.0, 4050.0, 4062.0]


def test_sizing_price_ignores_a_missing_next_settle_and_fills_holidays():
    idx = dates(4)
    es = continuous([4000.0, 4040.0], [np.nan, 40.0], idx[[0, 1]],
                    roll=[False, True], next_settle=[np.nan, np.nan])
    panel = settle_panel({"ES": es}, idx)
    assert panel["ES"].tolist() == [4000.0, 4040.0, 4040.0, 4040.0]


# --- calendars -------------------------------------------------------------

def test_a_product_keeps_its_position_on_its_own_holiday():
    idx = dates(4)
    es = continuous([4000.0] * 4, [np.nan, 0.0, 0.0, 0.0], idx)
    # NQ has no row on idx[2]; the target of 5 contracts that day cannot be traded.
    nq_idx = idx[[0, 1, 3]]
    nq = continuous([20000.0, 20100.0, 20300.0], [np.nan, 100.0, 200.0], nq_idx)
    res = run_backtest(book(idx, ES=[0, 0, 0, 0], NQ=[1, 1, 5, 1]), {"ES": es, "NQ": nq}, CFG)

    assert res["pos_NQ"].tolist() == [1, 1, 1, 1]
    assert res["contracts_traded"].tolist() == [1, 0, 0, 0]
    # idx[2]: no NQ move; idx[3]: the move spans the gap (200 points) on the 1 contract held.
    assert res["gross_pnl"].tolist() == [0.0, 2000.0, 0.0, 4000.0]


def test_a_late_starting_product_trades_nothing_before_its_first_row():
    idx = dates(4)
    es = continuous([4000.0] * 4, [np.nan, 10.0, 10.0, 10.0], idx)
    rty = continuous([2000.0, 2000.0], [np.nan, 5.0], idx[[2, 3]])
    res = run_backtest(book(idx, ES=[1, 1, 1, 1], RTY=[0, 0, 2, 2]), {"ES": es, "RTY": rty}, CFG)
    assert res["pos_RTY"].tolist() == [0, 0, 2, 2]
    assert res["gross_pnl"].tolist() == [0.0, 500.0, 500.0, 500.0 + 2 * 5.0 * 50]


# --- equity and margin -----------------------------------------------------

def test_returns_and_margin_usage_are_consistent_with_equity():
    idx = dates(4)
    es = continuous([4000.0, 4010.0, 4005.0, 4020.0], [np.nan, 10.0, -5.0, 15.0], idx)
    res = run_backtest(book(idx, ES=[2, 2, 2, 2]), {"ES": es}, CFG)
    assert res["net_pnl"].tolist() == pytest.approx((res["gross_pnl"] - res["costs"]).tolist())
    assert res["ret"].iloc[0] == pytest.approx(res["net_pnl"].iloc[0] / 1_000_000)
    assert res["ret"].iloc[1:].tolist() == pytest.approx(
        (res["net_pnl"] / res["equity"].shift(1)).iloc[1:].tolist())
    assert res["margin_usage"].tolist() == pytest.approx((res["margin"] / res["equity"]).tolist())
    assert (res["margin"] == 0.08 * res["gross_notional"]).all()


def test_no_positions_gives_an_empty_result():
    idx = dates(3)
    es = continuous([4000.0] * 3, [np.nan, 1.0, 1.0], idx)
    res = run_backtest(book(idx, ES=[0, 0, 0]), {"ES": es}, CFG)
    assert res.empty
    assert list(res.columns) == RESULT_COLUMNS + ["pos_ES"]


def test_missing_price_series_is_an_error():
    idx = dates(3)
    with pytest.raises(KeyError):
        run_backtest(book(idx, ES=[1, 1, 1]), {}, CFG)


# --- the whole pipeline ----------------------------------------------------

def test_signals_sizing_and_backtest_run_end_to_end():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    prices = {}
    for root, start in (("ES", 4000.0), ("NQ", 9000.0)):
        ret = pd.Series(0.0004 + 0.01 * rng.standard_normal(len(idx)), index=idx)
        ret.iloc[0] = np.nan
        settle = start * (1 + ret.fillna(0)).cumprod()
        prices[root] = pd.DataFrame({
            "settle": settle, "ret": ret, "pnl_points": settle.diff(), "adj_price": settle,
            "roll": False, "next_settle": np.nan})
    cfg = {**CFG,
           "signal": {"method": "time_series_momentum", "lookback_days": 60, "vol_lookback_days": 30},
           "sizing": {"method": "volatility_target", "capital": 10_000_000,
                      "target_annual_vol": 0.10, "max_gross_leverage": 3.0}}
    positions = size_positions(generate_signals(prices, cfg), prices, cfg)
    res = run_backtest(positions, prices, cfg)

    assert len(res) > 0 and res.index[0] > idx[59]          # starts after the warm-up
    assert res.notna().all().all()
    assert (res["costs"] >= 0).all() and (res["contracts_traded"] >= 0).all()
    assert res["equity"].iloc[-1] == pytest.approx(10_000_000 + res["net_pnl"].sum())
