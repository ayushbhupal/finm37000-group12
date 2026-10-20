"""Tests for issue #5: EDA helpers. Fake data only, no data files."""

import numpy as np
import pandas as pd
import pytest

from src import eda
from src.data import build_continuous

DATES = pd.bdate_range("2024-03-01", "2024-04-05")
RULE = {"rule": "days_before_expiry", "days_before_expiry": 5, "adjustment": "ratio_adjusted"}


def contract(symbol, expiry, settle):
    settle = np.asarray(settle, dtype="float64")
    return pd.DataFrame({
        "date": DATES, "symbol": symbol, "instrument_id": 1, "expiry": pd.Timestamp(expiry),
        "open": settle, "high": settle, "low": settle, "close": settle, "settle": settle,
        "volume": 1000, "cleared_volume": pd.array([1000] * len(DATES), dtype="Int64"),
        "open_interest": pd.array([5000] * len(DATES), dtype="Int64"),
    })


def two_contracts(front=100.0, nxt=110.0):
    n = len(DATES)
    return pd.concat([contract("ESH4", "2024-03-15", np.full(n, front)),
                      contract("ESM4", "2024-06-21", np.full(n, nxt))], ignore_index=True)


def random_returns(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"A": rng.normal(0.0005, 0.01, n), "B": rng.normal(0, 0.02, n)}, index=index)


def test_return_summary_scales_to_annual_figures():
    ret = pd.DataFrame({"A": [0.01, -0.01] * 50}, index=pd.bdate_range("2020-01-01", periods=100))
    out = eda.return_summary(ret)
    assert out.loc["A", "ann_return"] == pytest.approx(0.0)
    assert out.loc["A", "ann_vol"] == pytest.approx(ret["A"].std() * np.sqrt(252))
    assert out.loc["A", "days"] == 100


def test_return_summary_skips_leading_nans():
    ret = random_returns()
    ret.iloc[:100, 1] = np.nan
    assert eda.return_summary(ret).loc["B", "days"] == 900


def test_rolling_vol_needs_a_full_window():
    ret = random_returns()
    vol = eda.rolling_vol(ret, 60)
    assert vol["A"].iloc[:59].isna().all() and vol["A"].iloc[59:].notna().all()
    assert vol["A"].iloc[-1] == pytest.approx(ret["A"].iloc[-60:].std() * np.sqrt(252))


def test_rolling_vol_is_not_blanked_by_a_missing_date():
    ret = random_returns()
    ret.iloc[300, 0] = np.nan
    vol = eda.rolling_vol(ret, 60)
    assert np.isnan(vol["A"].iloc[300])
    assert vol["A"].iloc[301:].notna().all()


def test_missing_days_lists_dates_only_one_product_lacks():
    cont = build_continuous(two_contracts(), RULE)
    gap = cont.index[10]
    out = eda.missing_days({"ES": cont.drop(gap), "NQ": cont, "RTY": cont.iloc[5:]})
    assert out.to_dict("records") == [{"product": "ES", "date": gap}]   # RTY's earlier dates are not gaps


def test_autocorrelation_finds_alternating_returns():
    ret = pd.DataFrame({"A": [0.01, -0.01] * 100})
    acf = eda.autocorrelation(ret, [1, 2])
    assert acf.loc[1, "A"] < -0.9 and acf.loc[2, "A"] > 0.9


def test_autocorrelation_of_noise_is_inside_the_band():
    ret = random_returns(n=2000)
    acf = eda.autocorrelation(ret, [1, 2, 3])
    assert acf.abs().max().max() < 2 * eda.autocorrelation_band(2000)


def test_trailing_return_compounds():
    ret = pd.Series([0.10, 0.10, -0.50, 0.0])
    out = eda.trailing_return(ret, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.21)
    assert out.iloc[2] == pytest.approx(1.1 * 0.5 - 1)


def test_momentum_persistence_on_a_trending_series_is_all_wins():
    index = pd.bdate_range("2015-01-01", periods=600)
    ret = pd.DataFrame({"A": 0.001}, index=index)
    out = eda.momentum_persistence(ret, lookback=20, horizon=5)
    assert out.loc["A", "pct_long"] == 1.0
    assert out.loc["A", "hit_rate"] == 1.0
    assert out.loc["A", "strategy_mean"] == pytest.approx(1.001 ** 5 - 1)
    assert out.loc["A", "strategy_mean"] == pytest.approx(out.loc["A", "mean_fwd_all"])


def test_momentum_persistence_samples_non_overlapping_periods():
    ret = random_returns(n=500)
    out = eda.momentum_persistence(ret, lookback=60, horizon=21)
    assert out.loc["A", "periods"] <= (500 - 60) / 21 + 1


def test_term_structure_reads_the_curve_from_the_held_contract():
    prices = two_contracts(front=100.0, nxt=110.0)
    cont = build_continuous(prices, RULE)
    ts = eda.term_structure(prices, cont)
    first = ts.iloc[0]
    assert first["front"] == 100.0 and first["next"] == 110.0
    gap = (pd.Timestamp("2024-06-21") - pd.Timestamp("2024-03-15")).days
    assert first["roll_yield"] == pytest.approx(np.log(100 / 110) * 365 / gap)   # contango is negative
    assert (ts.loc[ts.index <= "2024-03-08", "front_expiry"] == pd.Timestamp("2024-03-15")).all()


def test_term_structure_backwardation_is_positive():
    prices = two_contracts(front=110.0, nxt=100.0)
    ts = eda.term_structure(prices, build_continuous(prices, RULE))
    assert (ts["roll_yield"].dropna() > 0).all()


def test_term_structure_has_no_next_contract_for_the_last_expiry():
    prices = two_contracts()
    ts = eda.term_structure(prices, build_continuous(prices, RULE))
    last = ts[ts["front_expiry"] == pd.Timestamp("2024-06-21")]
    assert not last.empty
    assert last["next"].isna().all() and last["roll_yield"].isna().all()


def test_roll_summary_counts_the_gap_at_each_roll():
    prices = two_contracts(front=100.0, nxt=110.0)
    out = eda.roll_summary(build_continuous(prices, RULE))
    assert out["rolls"] == 1
    assert out["mean_gap"] == pytest.approx(0.10)
    assert out["pct_contango"] == 1.0


def test_return_panel_aligns_products_on_date():
    prices = two_contracts()
    cont = build_continuous(prices, RULE)
    panel = eda.return_panel({"ES": cont, "NQ": cont.iloc[5:]})
    assert list(panel.columns) == ["ES", "NQ"]
    assert panel["NQ"].iloc[:5].isna().all()
