"""Tests for issue #3: raw data assembly. Fake frames only, no network or API key."""

import pandas as pd
import pytest

from src.config import load_config
from src import data as data_module
from src.data import (
    PRICE_COLUMNS,
    STAT_CLEARED_VOLUME,
    STAT_OPEN_INTEREST,
    STAT_SETTLEMENT,
    build_contract_prices,
    download_raw,
    daily_bars,
    latest_statistic,
    stat_dates,
    load_contract_prices,
    outright_definitions,
    summarize_prices,
    trade_date,
)


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def definitions():
    return pd.DataFrame({
        "ts_event": [utc("2024-01-02"), utc("2024-01-03"), utc("2024-01-03"), utc("2024-01-03")],
        "instrument_class": ["F", "F", "F", "S"],
        "instrument_id": [1, 1, 2, 99],
        "raw_symbol": ["ESH4", "ESH4", "ESM4", "ESH4-ESM4"],
        "activation": [utc("2022-03-18 13:30")] * 4,
        "expiration": [utc("2024-03-15 13:30")] * 2 + [utc("2024-06-21 13:30")] * 2,
    })


def hourly():
    # Sunday 17:00 Chicago (23:00 UTC) opens Monday's trade date; Monday 16:00 CT (22:00 UTC) is its last hour.
    ts = [utc("2024-01-07 23:00"), utc("2024-01-08 15:00"), utc("2024-01-08 21:00"),
          utc("2024-01-08 23:00"),   # 17:00 CT Monday: Tuesday's session
          utc("2024-01-08 21:00")]
    return pd.DataFrame({
        "ts_event": ts,
        "instrument_id": [1, 1, 1, 1, 99],
        "open": [100.0, 101.0, 103.0, 110.0, 5.0],
        "high": [101.0, 105.0, 104.0, 111.0, 5.0],
        "low": [99.0, 100.0, 102.0, 109.0, 5.0],
        "close": [100.5, 104.0, 103.5, 110.5, 5.0],
        "volume": [10, 20, 30, 40, 7],
    })


def statistics():
    def row(ts_event, ts_ref, iid, stat_type, price=float("nan"), quantity=0, flags=0):
        return {"ts_event": utc(ts_event), "ts_ref": utc(ts_ref), "instrument_id": iid,
                "stat_type": stat_type, "price": price, "quantity": quantity, "stat_flags": flags}
    return pd.DataFrame([
        row("2024-01-08 21:00", "2024-01-08", 1, STAT_SETTLEMENT, price=103.0),   # preliminary
        row("2024-01-09 00:03", "2024-01-08", 1, STAT_SETTLEMENT, price=103.25),  # final replaces it
        row("2024-01-09 14:00", "2024-01-08", 1, STAT_OPEN_INTEREST, quantity=500),
        row("2024-01-09 15:00", "2024-01-08", 1, STAT_OPEN_INTEREST, quantity=510),  # restated
        row("2024-01-09 14:00", "2024-01-08", 1, STAT_CLEARED_VOLUME, quantity=75),
    ])


def test_trade_date_rolls_at_1700_chicago():
    ts = pd.Series([utc("2024-01-07 23:00"),    # Sunday 17:00 CT -> Monday
                    utc("2024-01-08 22:59"),    # Monday 16:59 CT -> Monday
                    utc("2024-01-08 23:00"),    # Monday 17:00 CT -> Tuesday
                    utc("2024-07-08 21:59")])   # summer (CDT, UTC-5): Monday 16:59 CT -> Monday
    assert list(trade_date(ts)) == list(pd.to_datetime(["2024-01-08", "2024-01-08", "2024-01-09", "2024-07-08"]))


def test_outright_definitions_drop_spreads_and_duplicates():
    out = outright_definitions(definitions())
    assert sorted(out["symbol"]) == ["ESH4", "ESM4"]
    assert out.set_index("symbol")["expiry"]["ESH4"] == pd.Timestamp("2024-03-15")


def test_daily_bars_aggregate_by_trade_date():
    daily = daily_bars(hourly())
    mon = daily[(daily["instrument_id"] == 1) & (daily["date"] == "2024-01-08")].iloc[0]
    assert (mon["open"], mon["high"], mon["low"], mon["close"], mon["volume"]) == (100.0, 105.0, 99.0, 103.5, 60)
    tue = daily[(daily["instrument_id"] == 1) & (daily["date"] == "2024-01-09")].iloc[0]
    assert (tue["open"], tue["volume"]) == (110.0, 40)


def test_build_contract_prices_joins_everything_and_drops_spreads():
    prices = build_contract_prices(definitions(), hourly(), statistics())
    assert list(prices.columns) == PRICE_COLUMNS
    assert set(prices["symbol"]) == {"ESH4"}     # instrument 99 is a spread
    mon = prices[prices["date"] == "2024-01-08"].iloc[0]
    assert mon["settle"] == 103.25               # final settlement, not the preliminary one
    assert mon["open_interest"] == 510           # latest restatement
    assert mon["cleared_volume"] == 75
    assert mon["expiry"] == pd.Timestamp("2024-03-15")
    tue = prices[prices["date"] == "2024-01-09"].iloc[0]
    assert pd.isna(tue["settle"]) and pd.isna(tue["open_interest"])


def test_summarize_prices_counts_rows_and_missing_values():
    summary = summarize_prices(build_contract_prices(definitions(), hourly(), statistics()))
    row = summary.loc["ESH4"]
    assert row["rows"] == 2
    assert row["missing_settle"] == 1 and row["missing_open_interest"] == 1


def test_load_contract_prices_filters_dates_and_reports_missing_file(tmp_path):
    prices = build_contract_prices(definitions(), hourly(), statistics())
    prices.to_parquet(tmp_path / "ES_daily.parquet", index=False)
    got = load_contract_prices("es", "2024-01-09", "2024-01-31", raw_dir=tmp_path)
    assert list(got["date"]) == [pd.Timestamp("2024-01-09")]
    with pytest.raises(FileNotFoundError, match="python -m src.data"):
        load_contract_prices("NQ", "2024-01-01", "2024-01-31", raw_dir=tmp_path)


def test_config_has_data_section():
    data = load_config()["data"]
    assert data["dataset"] == "GLBX.MDP3"
    assert data["raw_dir"]


class BentoClientError(Exception):
    """Same class name as databento's, which is how the code recognises it."""


class FakeStore:
    def __init__(self, df):
        self.df = df

    def to_df(self, map_symbols=False):
        return self.df


class FakeClient:
    """Stands in for databento.Historical: serves the fixtures above and records requests."""

    def __init__(self):
        self.activation = utc("2023-01-01")
        self.unlisted = set()          # roots with no contracts, as RTY before 2017
        self.requests = []
        self.metadata = self
        self.timeseries = self

    def get_dataset_range(self, dataset):
        return {"start": "2010-06-06T00:00:00.000000000Z"}

    def get_range(self, dataset, symbols, stype_in, schema, start, end):
        self.requests.append({"schema": schema, "symbols": symbols, "stype_in": stype_in,
                              "start": start, "end": end})
        if schema == "definition":
            if symbols[0].split(".")[0] in self.unlisted:
                raise BentoClientError("422 symbology_invalid_request\nCould not resolve smart symbols")
            df = definitions().assign(activation=self.activation)
        elif schema == "ohlcv-1h":
            df = hourly()[hourly()["instrument_id"].isin([int(s) for s in symbols])]
        else:
            df = statistics()
        return FakeStore(df.set_index("ts_event"))


def test_download_raw_writes_one_parquet_per_product(tmp_path, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(data_module, "_client", lambda: fake)
    config = {"universe": ["ES"], "start_date": "2024-01-01", "end_date": "2024-01-31",
              "data": {"dataset": "GLBX.MDP3", "raw_dir": str(tmp_path)}}

    written = download_raw(config, workers=2)

    assert written == {"ES": tmp_path / "ES_daily.parquet"}
    prices = load_contract_prices("ES", "2024-01-01", "2024-01-31", raw_dir=tmp_path)
    assert list(prices.columns) == PRICE_COLUMNS
    assert set(prices["symbol"]) == {"ESH4"}
    # Hourly bars and statistics are requested for the outright contracts only (no spreads).
    stage2 = [r for r in fake.requests if r["schema"] != "definition"]
    assert stage2 and all(r["stype_in"] == "instrument_id" for r in stage2)
    assert all(set(r["symbols"]) == {"1", "2"} for r in stage2)

    # A second run is served from the cache: no new data requests.
    n = len(fake.requests)
    download_raw(config, workers=2)
    assert len(fake.requests) == n


class BentoServerError(Exception):
    """Same class name as databento's, so the retry logic treats it as a server error."""


def test_retry_recovers_from_server_errors_but_not_other_errors(monkeypatch):
    monkeypatch.setattr(data_module.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise BentoServerError("504 The remote gateway timed out.")
        return "ok"

    assert data_module._retry(flaky) == "ok" and calls["n"] == 3

    with pytest.raises(BentoServerError):
        data_module._retry(lambda: (_ for _ in ()).throw(BentoServerError("down")), attempts=2)
    with pytest.raises(ValueError):
        data_module._retry(lambda: (_ for _ in ()).throw(ValueError("bad request")))


def test_chunk_bounds_clamp_to_range_and_split_by_year():
    from src.data import _chunk_bounds
    start, end = pd.Timestamp("2010-06-06"), pd.Timestamp("2025-12-31")
    assert _chunk_bounds(2010, start, end) == (pd.Timestamp("2010-06-06"), pd.Timestamp("2011-01-01"))
    assert _chunk_bounds(2015, start, end) == (pd.Timestamp("2015-01-01"), pd.Timestamp("2016-01-01"))
    assert _chunk_bounds(2025, start, end) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"))
    short = _chunk_bounds(2025, pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-06"))
    assert short == (pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-07"))   # end is exclusive


def test_download_raw_clamps_start_and_requests_each_year(tmp_path, monkeypatch):
    fake = FakeClient()
    fake.activation = utc("2009-01-01")     # contracts already listed in 2010 and 2011
    monkeypatch.setattr(data_module, "_client", lambda: fake)
    config = {"universe": ["ES"], "start_date": "2010-01-01", "end_date": "2011-12-31",
              "data": {"dataset": "GLBX.MDP3", "raw_dir": str(tmp_path)}}
    download_raw(config, workers=2)

    definition = [r for r in fake.requests if r["schema"] == "definition"]
    assert len(definition) == 1                      # one definition request per product
    assert definition[0]["start"] == "2010-06-06"    # clamped to the first day Databento has
    bars = sorted(r["start"] for r in fake.requests if r["schema"] == "ohlcv-1h")
    assert bars == ["2010-06-06", "2011-01-01"]      # one request per year


def test_client_requires_api_key(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DATABENTO_API_KEY"):
        data_module._client()


def test_product_not_listed_in_range_is_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    fake = FakeClient()
    fake.unlisted = {"RTY"}
    monkeypatch.setattr(data_module, "_client", lambda: fake)
    config = {"universe": ["ES", "RTY"], "start_date": "2024-01-01", "end_date": "2024-01-31",
              "data": {"dataset": "GLBX.MDP3", "raw_dir": str(tmp_path)}}

    written = download_raw(config, workers=2)

    assert list(written) == ["ES"] and not (tmp_path / "RTY_daily.parquet").exists()
    assert "RTY: no contracts listed" in capsys.readouterr().out
    assert not [r for r in fake.requests if r["schema"] != "definition" and "RTY" in str(r["symbols"])]


def test_years_with_no_listed_contracts_make_no_data_requests(tmp_path, monkeypatch):
    fake = FakeClient()      # fixture contracts are listed from 2023, expire in 2024
    monkeypatch.setattr(data_module, "_client", lambda: fake)
    config = {"universe": ["ES"], "start_date": "2020-01-01", "end_date": "2024-01-31",
              "data": {"dataset": "GLBX.MDP3", "raw_dir": str(tmp_path)}}
    download_raw(config, workers=2)
    years = sorted({r["start"][:4] for r in fake.requests if r["schema"] == "ohlcv-1h"})
    assert years == ["2023", "2024"]


def stats_frame(rows):
    cols = ["ts_event", "ts_ref", "instrument_id", "stat_type", "price", "quantity", "stat_flags"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, format="ISO8601")
    df["ts_ref"] = pd.to_datetime(df["ts_ref"], utc=True, format="ISO8601")
    return df


def test_settlement_prefers_tick_rounded_record_even_if_published_later():
    # 2010-style: official 1043.75 (flag 4) is followed milliseconds later by unrounded 1043.70 (flag 0)
    stats = stats_frame([
        ("2010-06-07 21:21:15.508", "2010-06-07", 1, STAT_SETTLEMENT, 1043.75, 0, 4),
        ("2010-06-07 21:21:15.517", "2010-06-07", 1, STAT_SETTLEMENT, 1043.70, 0, 0),
        ("2024-02-02 00:03", "2024-02-01", 1, STAT_SETTLEMENT, 4928.50, 0, 3),   # 2024-style, no rounded flag:
        ("2024-02-01 21:01", "2024-02-01", 1, STAT_SETTLEMENT, 4928.25, 0, 2),   # the latest (final) one wins
    ])
    got = latest_statistic(stats, STAT_SETTLEMENT, "settle").set_index("date")["settle"]
    assert got[pd.Timestamp("2010-06-07")] == 1043.75
    assert got[pd.Timestamp("2024-02-01")] == 4928.50


def test_undated_open_interest_applies_to_previous_business_day():
    # Published Tuesday 2010-06-08 07:27 UTC (during Tuesday's trade date), no ts_ref -> describes Monday.
    # Published Monday 2010-06-14 07:27 UTC -> describes Friday 2010-06-11.
    stats = stats_frame([
        ("2010-06-08 07:27", pd.NaT, 1, STAT_OPEN_INTEREST, float("nan"), 300, 0),
        ("2010-06-08 13:40", pd.NaT, 1, STAT_OPEN_INTEREST, float("nan"), 310, 0),   # later revision wins
        ("2010-06-14 07:27", pd.NaT, 1, STAT_OPEN_INTEREST, float("nan"), 500, 0),
        ("2010-06-09 20:00", pd.NaT, 1, STAT_SETTLEMENT, 1043.75, 0, 4),             # undated settlement: dropped
    ])
    oi = latest_statistic(stats, STAT_OPEN_INTEREST, "open_interest").set_index("date")["open_interest"]
    assert oi[pd.Timestamp("2010-06-07")] == 310
    assert oi[pd.Timestamp("2010-06-11")] == 500
    assert latest_statistic(stats, STAT_SETTLEMENT, "settle").empty
    assert stat_dates(stats).notna().sum() == 3


def test_reused_instrument_id_does_not_leak_bars_onto_expired_contract():
    # Id 1 was ESH4 until it expired 2024-03-15, then Databento reused it for a calendar spread
    # whose bars (prices around 27) must not be attributed to ESH4.
    later = pd.DataFrame({
        "ts_event": [utc("2024-04-10 15:00"), utc("2022-01-05 15:00")],
        "instrument_id": [1, 1],
        "open": [27.0, 9.0], "high": [28.0, 9.0], "low": [26.0, 9.0], "close": [27.5, 9.0],
        "volume": [5, 5],
    })
    prices = build_contract_prices(definitions(), pd.concat([hourly(), later], ignore_index=True), statistics())
    h4 = prices[prices["symbol"] == "ESH4"]
    assert h4["date"].max() <= pd.Timestamp("2024-03-15")
    assert h4["date"].min() >= pd.Timestamp("2022-03-18")      # nor bars from before it was listed
    assert (h4["close"] > 50).all()


def test_active_ids_use_each_contracts_own_listing_window():
    from src.data import _active_ids
    d = definitions()
    assert _active_ids(d, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")) == [1, 2]
    assert _active_ids(d, pd.Timestamp("2024-04-01"), pd.Timestamp("2025-01-01")) == [2]    # ESH4 expired
    assert _active_ids(d, pd.Timestamp("2021-01-01"), pd.Timestamp("2022-01-01")) == []     # not yet listed


def test_repeated_definitions_with_shifted_listing_time_stay_one_contract():
    # Databento sometimes repeats a definition with the listing time a day apart (seen for YM).
    d = definitions()
    extra = d.iloc[[0]].assign(activation=utc("2022-03-19 13:30"))
    out = outright_definitions(pd.concat([d, extra], ignore_index=True))
    h4 = out[out["symbol"] == "ESH4"]
    assert len(h4) == 1 and h4.iloc[0]["listed"] == pd.Timestamp("2022-03-18")
    prices = build_contract_prices(pd.concat([d, extra], ignore_index=True), hourly(), statistics())
    assert not prices.duplicated(["instrument_id", "date"]).any()
