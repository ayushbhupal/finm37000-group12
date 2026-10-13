# Equity Futures Trading Strategy

A systematic trading strategy for equity index futures (e.g., E-mini S&P 500, Nasdaq-100), built as the team project for our Futures and Other Derivatives course.

## Goal

Design, backtest, and evaluate a rules-based strategy that trades equity index futures. By the end of the project, a user should be able to:

1. Load historical futures price data
2. Generate trading signals from a defined strategy
3. Run a backtest that accounts for realistic frictions (transaction costs, slippage, roll costs, margin)
4. View performance and risk results (returns, Sharpe ratio, drawdown, exposure, turnover)

## Strategy Overview

*Aspirational; the specifics will be refined as the team's research progresses.*

- **Universe:** Liquid equity index futures contracts
- **Signal:** To be decided (candidates: momentum / trend-following, mean reversion, carry or basis-based signals)
- **Position sizing:** Volatility-targeted sizing with leverage and margin limits
- **Contract rolling:** Systematic roll rule between expiries
- **Evaluation:** Out-of-sample testing and comparison against a buy-and-hold benchmark

## Setup

Requires Python 3.10+.

```bash
# 1. Clone the repository
git clone https://github.com/suniltrivedi7/finm37000-group12.git
cd finm37000-group12

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # makes `import src` work from notebooks and scripts

# 3. Add your Databento API key (only needed to download data)
cp .env.example .env             # then edit .env; it is git-ignored

# 4. Check everything works
pytest
```

## How to Run

*Each step is a placeholder until its issue is merged.*

```bash
python run_backtest.py --config config.yaml
# Results (metrics and plots) are written to ./results/
```

## Repository Structure

```
.
├── README.md
├── CONTRIBUTING.md     # branch / PR workflow
├── requirements.txt
├── pyproject.toml      # makes src/ an installable package
├── config.yaml         # strategy and backtest parameters
├── run_backtest.py     # single entry point
├── .env.example        # template for API keys (copy to .env)
├── data/
│   ├── raw/            # downloaded contract data (git-ignored)
│   └── processed/      # continuous series (git-ignored)
├── notebooks/          # exploration and EDA
├── src/
│   ├── contracts.py    # contract specs: multiplier, tick size, symbols
│   ├── config.py       # loads config.yaml
│   ├── data.py         # data loading and continuous contracts   (#3, #4)
│   ├── signals.py      # signal generation                       (#6)
│   ├── portfolio.py    # position sizing and risk limits         (#6)
│   ├── backtest.py     # backtest engine                         (#7)
│   └── analysis.py     # performance metrics and plots           (#8, #9)
├── tests/
├── reports/figures/    # figures for the final report
└── results/            # backtest output (git-ignored)
```

## Team

| Name | Role |
|------|------|
| Sunil Trivedi | Tech leader |
| Ayush Bhupal | Communication leader |
| Jon Nie | Design leader |

## Roadmap

Tasks are tracked in this repository's [Issues](../../issues). Each Issue is a piece of work that contributes to the goal above.

## Disclaimer

This is an academic project. Nothing here is investment advice, and backtested results do not guarantee future performance.
