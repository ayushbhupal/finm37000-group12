"""Signal generation.

Placeholder interface so other modules (and their tests) can import it now.
Implemented in issue #6.
"""

import pandas as pd


def generate_signals(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Return a signal per date and contract, using only data known at each date."""
    raise NotImplementedError("Issue #6: design signals and position sizing")
