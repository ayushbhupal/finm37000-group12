"""Performance metrics and charts.

Placeholder interface so other modules (and their tests) can import it now.
Implemented in issue #8 (metrics) and #9 (charts).
"""

import pandas as pd


def performance_summary(equity: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Return, volatility, Sharpe, Sortino, drawdown and turnover statistics."""
    raise NotImplementedError("Issue #8: performance metrics")


def make_figures(results: pd.DataFrame, output_dir: str) -> None:
    """Save the report figures to ``output_dir``."""
    raise NotImplementedError("Issue #9: generate charts and visualizations")
