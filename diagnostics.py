"""
Diagnostics — factor attribution & diversification sensitivity
==================================================================
Two questions this answers empirically instead of by guessing:

1. FACTOR ATTRIBUTION: of the factors that actually participate in the backtest
   (momentum, trend, seasonality -- fundamentals/analyst_sentiment/market_sentiment
   are mathematically inert in every backtest by design, see sector_rotation_model.py's
   lookahead-bias note), which ones have been helping vs. hurting since the backtest
   start date?

   Two complementary views:
   - LEAVE-ONE-OUT: run the full model with one factor's weight zeroed out (others
     renormalized). If CAGR goes UP without a factor, that factor was a net drag.
     If CAGR goes DOWN without it, that factor was net additive.
   - SINGLE-FACTOR: run the model with ONLY one factor driving the ranking (weight
     1.0, everything else 0). Shows how that factor performs standing alone, not just
     its marginal effect on the blend.

2. DIVERSIFICATION SENSITIVITY: a grid sweep over `top_n` (how many sectors held) and
   `max_weight` (concentration cap), holding everything else fixed, to see empirically
   whether the current diversification rules are costing meaningful performance in a
   monthly-rebalanced strategy -- rather than assuming.

Both reuse the SAME loaded price data across every variant (no extra network calls
beyond the initial price load), so this is CPU-bound, not I/O-bound, and safe to run
interactively behind a button in the dashboard.

Usage:
    from sector_rotation_model import SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG, DataProvider
    from diagnostics import run_leave_one_out, run_single_factor, run_concentration_sweep

    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider = DataProvider(tickers=tickers, start="2018-01-01")

    loo_df = run_leave_one_out(provider, CONFIG)
    sf_df = run_single_factor(provider, CONFIG)
    sweep_df = run_concentration_sweep(provider, CONFIG)
"""

from __future__ import annotations
import copy
import numpy as np
import pandas as pd

from sector_rotation_model import RotationBacktester, DataProvider


def _run_variant(provider: DataProvider, cfg: dict) -> dict:
    """Runs one backtest variant and extracts a compact stats row, including average
    monthly turnover (not part of the core stats dict) as an extra diagnostic."""
    bt = RotationBacktester(provider, cfg)
    result = bt.run()
    row = dict(result.stats)
    row["avg_monthly_turnover"] = float(result.turnover.mean()) if len(result.turnover) else np.nan
    row["n_rebalances"] = len(result.turnover)
    return row


def run_leave_one_out(provider: DataProvider, base_cfg: dict) -> pd.DataFrame:
    """For each factor, runs the model with that factor's weight zeroed out (the rest
    renormalize automatically -- CompositeScorer normalizes by whatever the weights sum
    to). Compares each variant's CAGR to the full baseline.

    Reading the output: `cagr_delta_if_removed` = baseline CAGR minus the leave-one-out
    CAGR. POSITIVE means removing the factor made things WORSE (baseline was higher) --
    i.e. the factor was net additive, keep/increase it. NEGATIVE means removing the
    factor made things BETTER -- i.e. the factor was a net drag, worth reducing or
    cutting.
    """
    baseline = _run_variant(provider, base_cfg)
    rows = {"baseline (all factors)": baseline}

    for factor in base_cfg["factor_weights"]:
        variant_cfg = copy.deepcopy(base_cfg)
        variant_cfg["factor_weights"] = dict(base_cfg["factor_weights"])
        variant_cfg["factor_weights"][factor] = 0.0
        if sum(variant_cfg["factor_weights"].values()) == 0:
            continue  # degenerate case: only one factor existed
        rows[f"without {factor}"] = _run_variant(provider, variant_cfg)

    df = pd.DataFrame(rows).T
    df["cagr_delta_if_removed"] = baseline["CAGR"] - df["CAGR"]
    df.loc["baseline (all factors)", "cagr_delta_if_removed"] = np.nan
    return df


def run_single_factor(provider: DataProvider, base_cfg: dict) -> pd.DataFrame:
    """For each factor, runs the model with ONLY that factor driving the ranking
    (weight 1.0, all others 0). Shows standalone performance -- useful alongside
    leave-one-out because a factor can look weak in combination but still be a
    genuinely strong standalone signal (or vice versa: a weak standalone factor can
    still add diversification value in the blend).

    CAVEAT: for fundamentals/analyst_sentiment/market_sentiment specifically (inert in
    every backtest -- see module docstring), isolating them as the ONLY factor means
    every sector ties at a score of 0. The resulting "CAGR" is whatever fixed portfolio
    falls out of that tie -- NOT a real standalone signal for those three. Expect their
    rows here to be identical to each other and not meaningfully interpretable; the
    leave-one-out view already proves they contribute nothing either way.
    """
    rows = {}
    for factor in base_cfg["factor_weights"]:
        variant_cfg = copy.deepcopy(base_cfg)
        variant_cfg["factor_weights"] = {f: (1.0 if f == factor else 0.0) for f in base_cfg["factor_weights"]}
        rows[factor] = _run_variant(provider, variant_cfg)
    return pd.DataFrame(rows).T


def run_concentration_sweep(
    provider: DataProvider,
    base_cfg: dict,
    top_n_values: list[int] = (2, 3, 4, 5, 6),
    max_weight_values: list[float] = (0.35, 0.50, 0.65, 1.00),
) -> pd.DataFrame:
    """Grid sweep over `top_n` (sectors held) x `max_weight` (concentration cap).
    For each combination, `min_sectors_held` and `hysteresis_rank` are set relative to
    `top_n` (min_sectors_held = top_n, hysteresis_rank = top_n + 2) so every cell is an
    internally consistent, comparable configuration rather than a fixed floor colliding
    awkwardly with a swept top_n.

    A max_weight of 1.00 means effectively uncapped (fully concentrated is allowed).
    """
    rows = []
    for top_n in top_n_values:
        for max_w in max_weight_values:
            variant_cfg = copy.deepcopy(base_cfg)
            variant_cfg["top_n"] = top_n
            variant_cfg["min_sectors_held"] = top_n
            variant_cfg["hysteresis_rank"] = top_n + 2
            variant_cfg["max_weight"] = max_w
            stats = _run_variant(provider, variant_cfg)
            rows.append({"top_n": top_n, "max_weight": max_w, **stats})
    return pd.DataFrame(rows)


def format_report(loo_df: pd.DataFrame, sf_df: pd.DataFrame, sweep_df: pd.DataFrame) -> str:
    """Plain-text summary for terminal/log output."""
    lines = []
    lines.append("=" * 70)
    lines.append("FACTOR ATTRIBUTION -- Leave-One-Out")
    lines.append("(positive cagr_delta_if_removed = factor is net ADDITIVE, keep it)")
    lines.append("(negative cagr_delta_if_removed = factor is a net DRAG, consider cutting)")
    lines.append("=" * 70)
    lines.append(loo_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_if_removed"]].to_string())
    lines.append("")
    lines.append("=" * 70)
    lines.append("FACTOR ATTRIBUTION -- Single-Factor Standalone")
    lines.append("=" * 70)
    lines.append(sf_df[["CAGR", "Sharpe", "MaxDrawdown"]].sort_values("CAGR", ascending=False).to_string())
    lines.append("")
    lines.append("=" * 70)
    lines.append("DIVERSIFICATION SENSITIVITY -- top_n x max_weight")
    lines.append("=" * 70)
    pivot = sweep_df.pivot(index="top_n", columns="max_weight", values="CAGR")
    lines.append(pivot.to_string())
    return "\n".join(lines)
