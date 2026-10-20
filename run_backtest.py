"""Single entry point: data -> signals -> positions -> backtest -> results.

Usage:
    python run_backtest.py --config config.yaml

Each step raises NotImplementedError until its issue is merged.
"""

import argparse

from src import analysis, backtest, data, portfolio, signals
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    prices = {
        root: data.build_continuous(
            data.load_contract_prices(root, config["start_date"], config["end_date"]),
            config["roll"],
        )
        for root in config["universe"]
    }
    sig = signals.generate_signals(prices, config)
    positions = portfolio.size_positions(sig, prices, config)
    results = backtest.run_backtest(positions, prices, config)
    print(analysis.performance_summary(results["equity"]))
    analysis.make_figures(results, config["output_dir"])


if __name__ == "__main__":
    main()
