# Strategy Research (Issue #2)

**Decision (2026-10-13):** the team voted unanimously for **time-series momentum with
volatility targeting**. The core universe stays ES, NQ, RTY and YM on daily data. A few
international index futures (e.g., FESX, NKD) will be added in issue #10 as a robustness
check, which would also make cross-sectional momentum testable later. Mean reversion and
carry / calendar spreads are deferred as possible extensions.

The write-ups below were drafted to support the vote, one per candidate strategy.
Each write-up covers the hypothesis, data needed, expected edge and key risks.
Claims about expected edge are qualitative priors from the academic and practitioner
literature, not results from our own data. Issue #5 (EDA) and the backtest will test them.

**Project context.** Universe is ES, NQ, RTY, YM (`config.yaml`), daily bars from 2010 to
2025, volatility-targeted sizing, costs of $2.50 commission plus 1 tick slippage per
contract. Data comes from Databento (CME Globex), which also fixes what we can build cheaply.

## 1. Time-series momentum / trend following

**Hypothesis.** An asset's own recent return predicts its near-future return. Investors
underreact to information and herd, so trends persist over horizons of roughly 1 to 12 months.
Each contract is long when its trailing return (or a moving-average crossover) is positive and
short when negative, independently of the others.

**Data needed.** Daily continuous close series per contract (issue #4), plus a trailing
volatility estimate for sizing. No open interest or term-structure data is required beyond what
the roll rule uses.

**Expected edge.** The most documented futures anomaly (Moskowitz, Ooi and Pedersen, 2012;
managed-futures/CTA literature). Across a diversified futures universe it has historically
delivered a positive Sharpe ratio. Within equity indices alone, the edge is weaker and less
diversified because the four contracts are highly correlated (about 0.8 to 0.95 daily). It
tends to help in prolonged drawdowns (2008, 2022-style) and lag in sharp V-shaped recoveries.

**Key risks.**
- Four highly correlated contracts mean effectively one bet, so results are noisy.
- Whipsaw losses in range-bound markets; sharp reversals (e.g., March 2020).
- Equity indices have a strong long-run upward drift, so a long-biased trend rule can look good
  mainly by being long. The buy-and-hold benchmark comparison is essential.
- Lookback choice is a tuning parameter and an overfitting risk (addressed in issue #10).

**Implementation effort:** Low. Fits the `generate_signals` and `size_positions` interfaces directly.

## 2. Mean reversion (short-horizon reversal / intraday)

**Hypothesis.** Short-term price moves in index futures partly reverse because of liquidity
provision, overreaction and index-arbitrage flows. Buy after a sharp short-horizon decline (for
example a multi-day low or an oversold RSI reading) and sell after a sharp rise.

**Data needed.** Daily bars suffice for multi-day reversal. Intraday versions need minute or
tick bars, which means much larger Databento downloads and cost, plus more careful handling of
sessions, rolls and execution timing.

**Expected edge.** Short-horizon equity index reversal has historically been strong at daily to
weekly horizons, particularly after large down days. However, the edge per trade is small, so
net performance is dominated by transaction costs and slippage.

**Key risks.**
- High turnover, so the result is very sensitive to our cost assumptions (1 tick of slippage may
  be optimistic or pessimistic depending on the time of day).
- Negative skew: many small gains and occasional large losses when a decline turns into a trend
  (2008, 2020). Buying into a falling market needs hard risk limits.
- Intraday variants raise data volume, latency and fill-realism concerns that daily backtests
  cannot represent.

**Implementation effort:** Low for daily, high for intraday.

## 3. Carry / roll-yield / calendar spreads

**Hypothesis.** Futures price equals spot grown at the financing rate less expected dividends,
so the gap between the front and next contract reflects the cost of carry. When the futures
curve slope (basis) is unusually steep or flat, the position earns or loses roll yield.
Strategies either hold outright positions in the high-carry contracts or trade the calendar
spread (long one expiry, short the next).

**Data needed.** Prices for at least the first two listed expiries per product on every date,
not only the front-month continuous series. Ideally also a financing rate and a dividend
estimate to separate carry from noise. Databento supplies the individual contracts, but
issues #3 and #4 would have to keep the second contract available.

**Expected edge.** Carry is a well-documented return source across asset classes (Koijen, Moskowitz,
Pedersen and Vrugt, 2018). For equity index futures, though, carry is mostly determined by
interest rates minus dividend yield, so it varies slowly and is similar across the four
contracts. Calendar-spread trading is closer to relative-value arbitrage, where edge is small
and competed away by faster participants.

**Key risks.**
- Little cross-sectional or time variation in carry, so the signal may be nearly constant in
  the 2010 to 2025 sample (rates were near zero for much of it).
- Spreads move in small increments, so costs are large relative to the expected gain.
- Roll mechanics are the strategy itself, so errors in expiry handling directly corrupt results.
- Needs more data wrangling than the other candidates.

**Implementation effort:** Medium to high.

## 4. Cross-sectional momentum across index futures

**Hypothesis.** Among the contracts in the universe, past winners keep outperforming past
losers. Rank contracts by trailing return, go long the top ranked and short the bottom ranked,
typically dollar- or volatility-neutral.

**Data needed.** Same daily continuous series as strategy 1. Broadening the universe to
international indices (for example Euro Stoxx 50, DAX, Nikkei, FTSE) would need data from other
venues (Eurex, ICE Europe, etc.), so check availability and cost on Databento before committing.

**Expected edge.** Cross-sectional momentum across country equity indices has been documented
(Asness, Moskowitz and Pedersen, 2013). With only ES, NQ, RTY and YM, the cross-section is
tiny and the contracts share one US-equity factor, so the long/short spread mostly captures
growth versus value or large versus small-cap rotations (NQ vs YM, RTY vs ES). That is a real but
thin and unstable edge. The market-neutral construction does remove the equity drift, which
makes it a cleaner test than a long-biased trend rule.

**Key risks.**
- With four assets, ranks are noisy and a single contract can dominate the result.
- Style factor reversals (for example the 2020 to 2021 value rotation) can cause sharp losses.
- Market neutral by construction means lower raw return, so leverage is needed to hit the 10%
  vol target, which raises margin and cost concerns.
- Extending to international contracts brings currency exposure, different trading hours and
  possibly unavailable data.

**Implementation effort:** Low to medium on the US universe, high with international contracts.

## 5. Other ideas (for the team to add)

Candidates the team could raise. None is fleshed out yet.

- **Trend plus volatility filter.** Strategy 1 with exposure scaled down when realized volatility
  is high, targeting the crash risk in the trend strategy.
- **Combination or ensemble.** Blend trend and short-horizon reversal, whose returns tend to be
  negatively correlated, which could improve Sharpe ratio and drawdown at the cost of more
  parameters.
- **Overnight vs intraday drift.** Hold only the overnight session (documented historical
  pattern in equity index returns). Requires intraday data and should be checked against
  recent-sample decay.

## Comparison

| | Trend (TSMOM) | Mean reversion | Carry / spreads | Cross-sectional momentum |
|---|---|---|---|---|
| Data cost | Low | Low (daily) / High (intraday) | Medium (needs 2nd expiry) | Low (US) / Medium-High (intl) |
| Turnover | Low | High | Medium | Medium |
| Cost sensitivity | Low | High | High | Medium |
| Diversification in a 4-contract universe | Poor | Poor | Poor | Poor (but market-neutral) |
| Overfit risk | Medium | Medium-High | Medium | Medium |
| Fit with `config.yaml` / vol-targeting | Direct | Direct | Needs changes | Direct |
| Effort | Low | Low / High | Medium-High | Low-Medium |

## Suggested discussion points

1. The four US contracts are highly correlated, so no strategy gets much diversification.
   Should we widen the universe (for example Micro contracts add nothing; international indices
   would) or accept it and focus on a clean, well-tested single strategy?
2. Daily or intraday? Daily keeps data and compute manageable and matches the current stub
   interfaces, and is probably the right scope for a course project.
3. What is the benchmark? A long-only buy-and-hold on the same contracts, vol-matched, so a
   long-biased strategy gets no free credit for equity drift.
4. Our writeup-level recommendation, pending the vote: time-series momentum with volatility
   targeting as the primary strategy (lowest cost, cleanest fit, best documented), with
   cross-sectional momentum or a trend plus reversal blend as a possible extension for the
   robustness work in issue #10. This is an opinion for the team to challenge, not a decision.

## References

- Moskowitz, Ooi, Pedersen (2012), "Time Series Momentum", Journal of Financial Economics.
- Asness, Moskowitz, Pedersen (2013), "Value and Momentum Everywhere", Journal of Finance.
- Koijen, Moskowitz, Pedersen, Vrugt (2018), "Carry", Journal of Financial Economics.
