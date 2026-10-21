"""Backtest engine (issue #7).

Turns whole-contract target positions into daily P&L, costs, equity and margin.

Timing
------
A position dated ``t`` is set at the settle of ``t`` and earns P&L on the next
trading date, so nothing is earned on the day a position is set. Trades fill at
the settle with no price impact; slippage is charged as a cost instead.

Each product trades only on its own calendar. On a date a product has no row
(its holiday or a gap in the source data) it keeps its previous position, and a
target dated that day is ignored. The next row's ``pnl_points`` already spans the gap.

Costs
-----
Every contract bought or sold pays ``costs.commission_per_contract`` plus
``costs.slippage_ticks`` ticks of slippage. A roll trades **both legs**: the old
contract is closed at ``settle`` and the new one opened at ``next_settle``, so a
roll day trades ``|previous| + |new|`` contracts, not ``|new - previous|``.

Equity and margin
-----------------
Equity is ``sizing.capital`` plus cumulative net P&L. Sizing stays on the fixed
initial capital (no compounding); daily returns are on the previous equity.
Margin is ``margin.pct_of_notional`` of gross notional. It is reported, but no
margin calls or forced liquidation are modelled, and cash earns no interest.
"""

import pandas as pd

from .contracts import get_spec
from .portfolio import settle_panel

RESULT_COLUMNS = ["equity", "ret", "gross_pnl", "costs", "net_pnl", "contracts_traded",
                  "gross_notional", "margin", "margin_usage"]


def _panel(prices: dict[str, pd.DataFrame], column: str, index: pd.Index, fill) -> pd.DataFrame:
    """One column of each product's continuous series on ``index``; missing dates get ``fill``."""
    return pd.DataFrame({root: df[column] for root, df in prices.items()}).reindex(index).fillna(fill)


def run_backtest(positions: pd.DataFrame, prices: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    """Daily P&L, costs, equity and margin for the target positions.

    ``positions`` is the output of ``portfolio.size_positions`` (whole contracts,
    one column per product, dated as in the module docstring). ``prices`` maps root
    to the output of ``data.build_continuous``. The result is indexed by date from
    the first date any position is held (the warm-up before it is dropped), with
    these columns:

    * ``equity``: USD equity at the settle, after that day's costs
    * ``ret``: net return on the previous equity
    * ``gross_pnl``, ``costs``, ``net_pnl``: daily USD
    * ``contracts_traded``: contracts bought or sold, both roll legs included
    * ``gross_notional``: sum of absolute position notional after the day's trades
    * ``margin``, ``margin_usage``: USD and share of equity
    * ``pos_<ROOT>``: contracts held after the day's trades, per product
    """
    roots = list(positions.columns)
    missing = [root for root in roots if root not in prices]
    if missing:
        raise KeyError(f"No price series for {missing}; got {sorted(prices)}")
    prices = {root: prices[root] for root in roots}
    index = positions.index
    specs = {root: get_spec(root) for root in roots}
    multiplier = pd.Series({root: spec.multiplier for root, spec in specs.items()})

    # A product only trades on dates it has a row; otherwise it keeps the previous position.
    tradable = pd.DataFrame({root: index.isin(prices[root].index) for root in roots}, index=index)
    held = positions.where(tradable).ffill().fillna(0).astype("int64")
    previous = held.shift(1, fill_value=0)

    gross_pnl = (previous * _panel(prices, "pnl_points", index, 0.0) * multiplier).sum(axis=1)

    on_roll = _panel(prices, "roll", index, False).astype(bool)
    traded = (held - previous).abs().where(~on_roll, held.abs() + previous.abs())
    per_contract = pd.Series({
        root: config["costs"]["commission_per_contract"]
        + config["costs"]["slippage_ticks"] * spec.tick_value
        for root, spec in specs.items()})
    costs = (traded * per_contract).sum(axis=1)

    notional = (held.abs() * settle_panel(prices, index) * multiplier).sum(axis=1)
    margin = notional * config["margin"]["pct_of_notional"]

    net_pnl = gross_pnl - costs
    capital = float(config["sizing"]["capital"])
    equity = capital + net_pnl.cumsum()
    result = pd.DataFrame({
        "equity": equity,
        "ret": net_pnl / equity.shift(1, fill_value=capital),
        "gross_pnl": gross_pnl, "costs": costs, "net_pnl": net_pnl,
        "contracts_traded": traded.sum(axis=1),
        "gross_notional": notional,
        "margin": margin, "margin_usage": margin / equity,
    })[RESULT_COLUMNS]
    result = result.join(held.add_prefix("pos_"))

    in_market = held.ne(0).any(axis=1)
    if not in_market.any():
        return result.iloc[0:0]
    return result.loc[in_market.idxmax():]
