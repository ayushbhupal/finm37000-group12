# Raw data dictionary (issue #3)

`python -m src.data --config config.yaml` writes one Parquet file per product to
`data/raw/<ROOT>_daily.parquet` (`ES`, `NQ`, `RTY`, `YM`). Each row is **one outright
futures contract on one CME trade date**, so a single date has several rows (one per
listed expiry). `data.load_contract_prices(root, start, end)` reads these files.

Raw files are git-ignored. Re-running the command reuses the per-year cache in
`data/raw/_cache/` and does not re-download; `--force` refreshes it.

## Source

Databento, dataset `GLBX.MDP3` (CME Globex), parent symbology (`ES.FUT` etc., every
expiry). Three schemas are combined, because no single one has everything:

| Schema | Used for |
|---|---|
| `ohlcv-1h` | open, high, low, close, volume (hourly bars, aggregated to daily here) |
| `definition` | expiry date; also identifies outright contracts so calendar spreads can be dropped |
| `statistics` | settlement price (stat type 3), cleared volume (6), open interest (9) |

Databento's own `ohlcv-1d` bars are **not** used. They are cut at 00:00 UTC, which splits
each CME session in two: the Sunday-evening hour becomes a separate "Sunday" bar, and
each bar's close falls after the next session has already reopened. Daily bars are
instead built from hourly bars by CME trade date (see `date` below).

## Columns

| Column | Type | Description |
|---|---|---|
| `date` | datetime64 | CME **trade date**. A session opens at 17:00 Chicago time the evening before, so Sunday-evening trading belongs to Monday. Weekdays only. |
| `symbol` | string | Exchange symbol, e.g. `ESH4` (H = March, M = June, U = September, Z = December; last digit = year). **Not unique across decades** (`ESH1` is 2011 and 2021); use `instrument_id` or `expiry` to identify a contract. |
| `instrument_id` | uint32 | Databento's numeric contract id. Unique per contract; the safest join key. |
| `expiry` | datetime64 | Last trading day, from the contract definition (UTC date). Equity index futures expire on the third Friday of the quarterly month, except when
  that day is an exchange holiday (for example `ESM7` expires Thursday 2027-06-17, because
  Friday the 18th is the observed Juneteenth holiday). All expiries in the 2025 sample are third Fridays. |
| `open` | float | First trade of the trade date (index points). |
| `high` | float | Highest trade of the trade date. |
| `low` | float | Lowest trade of the trade date. |
| `close` | float | Price of the last hourly bar of the trade date (about 16:00 Chicago). This is **not** the settlement price. |
| `settle` | float | Official CME daily settlement price (the exchange's tick-rounded value). Use this, not `close`, for marking positions and returns. Missing on exchange holidays (see below). |
| `volume` | int64 | Contracts traded in the bars, summed over the trade date. Counts the contract's own order book only. |
| `cleared_volume` | Int64 | Official cleared volume from the exchange. Larger than `volume` because it also counts spread legs and off-book trades (typically 5–8% more for ES). Nullable. Databento reported **0** for trade date 2025-06-05 on every product (a source glitch: traded `volume` was normal), so treat `cleared_volume == 0` with `volume > 0` as missing. |
| `open_interest` | Int64 | Open contracts at the end of the trade date. Nullable. |

All prices are in index points. Multiply by the contract multiplier in `src/contracts.py`
to get dollars. Prices are the raw, unadjusted prices of each individual contract:
**no roll adjustment** is applied here (that is issue #4).

## Things to know before using the data

- **Open interest is published the morning after.** The value for trade date D is first
  available on D+1. A signal computed at the close of D must not use D's open interest
  (look-ahead). Lag it by one day, which matters for any open-interest roll rule in #4.
- **Settlement and open interest are final values.** Both are revised after first
  publication (preliminary then final settlement; restated open interest). The latest
  record for each contract and date is kept. Databento also publishes an unrounded
  settlement next to the official tick-rounded one (e.g. 1043.70 beside 1043.75); the
  rounded record (flag bit 4) is preferred even if the unrounded one arrives later.
- **Older open interest and cleared volume have no date.** In the 2010 and 2013 samples
  checked, these records carry no `ts_ref`. They are published during trade date T and
  describe the previous business day, so that is the date assigned (the implied
  cleared-to-traded volume ratio moves smoothly day to day, as the dated data does).
  This rule was checked on sample weeks only.
- **Exchange holidays.** On days such as Martin Luther King Day and Presidents' Day,
  CME equity futures trade a short session and publish no settlement, open interest or
  cleared volume. Those rows have `settle`, `cleared_volume` and `open_interest` empty.
  Drop them or forward-fill when building continuous series (#4).
- **Expiry day.** Open interest can be missing on the expiry date itself (seen for `ESH4` on 2024-03-15).
- **Start date.** `config.yaml` asks for 2010-01-01, but `GLBX.MDP3` data begins
  2010-06-06, so the earliest rows are from June 2010.
- **RTY starts in 2017.** Russell 2000 futures only exist in Databento's CME data from
  2017-06-04. ES, NQ and YM go back to 2010. A range with no RTY contracts skips RTY with a
  message; a longer range simply has RTY rows from mid-2017.
- **Thin contracts.** Back-month contracts list years ahead of expiry and trade a few
  contracts a day, so their prices are unreliable. Roll logic should only look at the
  front few expiries.
- **Data quality flags from Databento.** A partial pull of 2010–2014 ES data warned that
  some days were "degraded" in the source (2014-06-11 to 2014-06-13 and possibly
  others). Check these dates for gaps before relying on them.

## Coverage

The full configured range (`config.yaml`: 2010-01-01 to 2025-12-31, clamped to Databento's
first available day, 2010-06-06) has **not been downloaded yet**; only the small sample below
has been run to check the pipeline. Running `python -m src.data --config config.yaml` with
no overrides downloads everything; cached chunks are reused. A shorter range can be pulled
with `--start` / `--end`, for example `--start 2025-06-02 --end 2025-06-06` for the sample.

Sample run, 2025-06-02 to 2025-06-06 (5 trade dates), all four products:

| Product | Rows | Contracts | Rows per contract | Missing `settle` | Missing `open_interest` |
|---|---|---|---|---|---|
| ES | 15 | `ESM5`, `ESU5`, `ESZ5` | 5, 5, 5 | 0 | 0 |
| NQ | 15 | `NQM5`, `NQU5`, `NQZ5` | 5, 5, 5 | 0 | 0 |
| RTY | 12 | `RTYM5`, `RTYU5`, `RTYZ5` | 5, 5, 2 | 0 | 0 |
| YM | 15 | `YMM5`, `YMU5`, `YMZ5` | 5, 5, 5 | 0 | 0 |

Expiries are 2025-06-20 (M), 2025-09-19 (U) and 2025-12-19 (Z), all third Fridays. `RTYZ5` has
only 2 rows because it barely traded that week. Dates are Monday to Friday only. For the
front contract, `|close - settle| / settle` is at most 0.3%.
