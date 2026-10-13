"""Position sizing and risk limits.

Placeholder interface so other modules (and their tests) can import it now.
Implemented in issue #6.
"""

import pandas as pd


def size_positions(signals: pd.DataFrame, prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Convert signals into whole-contract target positions under the risk limits."""
    raise NotImplementedError("Issue #6: design signals and position sizing")
