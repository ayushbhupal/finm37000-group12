"""Tests for issue #10: out-of-sample helpers. Fake returns only, no data files."""

import numpy as np
import pandas as pd
import pytest

from src import robustness

INDEX = pd.bdate_range("2011-06-01", "2021-12-31")


def noise(mean, seed, sd=0.01, index=INDEX):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, len(index)), index=index)


def test_variants_are_every_combination():
    out = robustness.variants({"lookback_days": [63, 252], "skip_days": [0, 21]})
    assert len(out) == 4
    assert {"lookback_days": 252, "skip_days": 21} in out


def test_label_names_the_parameters():
    assert robustness.label({"lookback_days": 252, "skip_days": 0}) == "lookback_days=252, skip_days=0"


def test_sharpe_scales_to_a_year_and_handles_series_and_frames():
    r = pd.Series([0.01, 0.0, 0.02, -0.01] * 50)
    expected = r.mean() / r.std() * np.sqrt(252)
    assert robustness.sharpe(r) == pytest.approx(expected)
    assert robustness.sharpe(pd.DataFrame({"a": r, "b": -r})).tolist() == pytest.approx([expected, -expected])


def test_sharpe_of_a_flat_series_is_nan():
    assert np.isnan(robustness.sharpe(pd.Series([0.0] * 20)))


def test_sharpe_ignores_nan():
    r = pd.Series([np.nan] * 10 + [0.01, -0.01, 0.02, 0.0] * 10)
    assert robustness.sharpe(r) == pytest.approx(robustness.sharpe(r.dropna()))


def test_split_puts_the_cut_date_out_of_sample():
    r = noise(0, 0)
    before, after = robustness.split(r, "2019-01-01")
    assert before.index.max() < pd.Timestamp("2019-01-01") <= after.index.min()
    assert len(before) + len(after) == len(r)


def test_in_vs_out_reports_both_periods_and_ranks():
    grid = pd.DataFrame({"good": noise(0.002, 1), "bad": noise(-0.002, 2)})
    table = robustness.in_vs_out(grid, "2019-01-01")
    assert table.loc["good", "sharpe_in"] > 0 > table.loc["bad", "sharpe_in"]
    assert table.loc["good", "rank_out"] == 1


def test_rank_correlation_is_one_when_rankings_agree_and_minus_one_when_reversed():
    agree = pd.DataFrame({"sharpe_in": [0.1, 0.5, 0.9], "sharpe_out": [0.0, 0.2, 0.3]})
    assert robustness.rank_correlation(agree) == pytest.approx(1.0)
    assert robustness.rank_correlation(agree.assign(sharpe_out=[0.3, 0.2, 0.0])) == pytest.approx(-1.0)


def test_walk_forward_picks_using_only_earlier_years():
    # "late" only becomes the best variant from 2018; walk-forward must not pick it before then.
    early = noise(0.003, 3)
    late = noise(0.0, 4)
    late[late.index.year >= 2018] += 0.02
    grid = pd.DataFrame({"early": early, "late": late})
    out, chosen = robustness.walk_forward(grid, first_test_year=2017, min_train_years=5)
    assert chosen.loc[2017, "variant"] == "early"
    assert chosen.loc[2018, "variant"] == "early"     # 2018's own returns are not yet known
    assert chosen.loc[2021, "variant"] == "late"
    assert out.index.min() >= pd.Timestamp("2017-01-01") and out.index.max().year == 2021


def test_walk_forward_skips_years_without_enough_history():
    grid = pd.DataFrame({"a": noise(0.001, 5)})
    out, chosen = robustness.walk_forward(grid, first_test_year=2013, min_train_years=5)
    assert list(chosen.index) == list(range(2017, 2022))    # first year with 5 years behind it
    assert out.index.min().year == 2017


def test_walk_forward_ignores_a_variant_that_has_not_started():
    a = noise(0.001, 6)
    b = noise(0.01, 7)
    b[b.index.year < 2016] = np.nan                     # starts late, so only a few training years
    out, chosen = robustness.walk_forward(pd.DataFrame({"a": a, "b": b}), 2017, 5)
    assert not out.isna().any()


def test_load_or_run_grid_reuses_the_cache_until_the_config_changes(tmp_path, monkeypatch):
    calls = []

    def fake_run_grid(prices, config, grid):
        calls.append(1)
        return pd.DataFrame({"a": [0.01, 0.02]}, index=pd.bdate_range("2020-01-01", periods=2))

    monkeypatch.setattr(robustness, "run_grid", fake_run_grid)
    config = {"signal": {"lookback_days": 252}, "robustness": {"grid": {"lookback_days": [63]}}}
    path = tmp_path / "grid.parquet"
    first = robustness.load_or_run_grid({}, config, path)
    again = robustness.load_or_run_grid({}, config, path)
    assert len(calls) == 1 and first.equals(again)
    config["robustness"]["grid"]["lookback_days"] = [63, 126]       # a different grid
    robustness.load_or_run_grid({}, config, path)
    assert len(calls) == 2
