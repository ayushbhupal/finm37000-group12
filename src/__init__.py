"""Equity index futures trading strategy (FINM 37000, group 12).

Modules
-------
contracts  Contract specifications (multiplier, tick size, expiry cycle).
config     Loads strategy and backtest parameters from ``config.yaml``.
data       Data download and continuous-contract construction (issues #3, #4).
signals    Signal generation (issue #6).
portfolio  Position sizing and risk limits (issue #6).
backtest   Backtest engine (issue #7).
analysis   Performance metrics and charts (issues #8, #9).
"""

__version__ = "0.1.0"
