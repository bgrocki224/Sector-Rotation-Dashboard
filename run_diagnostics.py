"""
Run Diagnostics — terminal entry point
=========================================
Runs factor attribution (leave-one-out + single-factor) and a diversification
sensitivity sweep (top_n x max_weight) against live price data, prints a report,
and saves the underlying tables as CSVs for further analysis.

Usage:
    python3 run_diagnostics.py
    python3 run_diagnostics.py --start 2018-01-01
    python3 run_diagnostics.py --start 2018-01-01 --top-n 2 3 4 5 6 --max-weight 0.35 0.5 0.65 1.0

Takes a few minutes on first run (price download); subsequent backtest variants are
CPU-only and fast since they reuse the same cached price data.
"""

import argparse
import sys

from sector_rotation_model import SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG, DataProvider
from diagnostics import run_leave_one_out, run_single_factor, run_concentration_sweep, format_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--top-n", nargs="+", type=int, default=[2, 3, 4, 5, 6],
                         help="top_n values to sweep")
    parser.add_argument("--max-weight", nargs="+", type=float, default=[0.35, 0.50, 0.65, 1.00],
                         help="max_weight values to sweep")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip the concentration sweep (faster)")
    args = parser.parse_args()

    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider = DataProvider(tickers=tickers, start=args.start)

    print(f"Loading price data from {args.start}...", file=sys.stderr)
    provider.load_prices()  # warm the cache once, up front

    print("Running leave-one-out attribution...", file=sys.stderr)
    loo_df = run_leave_one_out(provider, CONFIG)
    loo_df.to_csv("attribution_leave_one_out.csv")

    print("Running single-factor attribution...", file=sys.stderr)
    sf_df = run_single_factor(provider, CONFIG)
    sf_df.to_csv("attribution_single_factor.csv")

    if args.skip_sweep:
        sweep_df = None
    else:
        print("Running concentration sensitivity sweep...", file=sys.stderr)
        sweep_df = run_concentration_sweep(provider, CONFIG, args.top_n, args.max_weight)
        sweep_df.to_csv("sensitivity_sweep.csv")

    print()
    if sweep_df is not None:
        print(format_report(loo_df, sf_df, sweep_df))
    else:
        print("=" * 70)
        print("FACTOR ATTRIBUTION -- Leave-One-Out")
        print("=" * 70)
        print(loo_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_if_removed"]].to_string())
        print()
        print("=" * 70)
        print("FACTOR ATTRIBUTION -- Single-Factor Standalone")
        print("=" * 70)
        print(sf_df[["CAGR", "Sharpe", "MaxDrawdown"]].sort_values("CAGR", ascending=False).to_string())
    print()
    print("Saved: attribution_leave_one_out.csv, attribution_single_factor.csv"
          + ("" if args.skip_sweep else ", sensitivity_sweep.csv"))


if __name__ == "__main__":
    main()
