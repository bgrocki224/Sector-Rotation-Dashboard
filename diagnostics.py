"""
Diagnostics — factor attribution & diversification sensitivity
==================================================================
Two questions this answers empirically instead of by guessing:

1. FACTOR ATTRIBUTION: of the factors that actually participate in the backtest
   (momentum, trend, seasonality, macro_regime and the RSI overbought overlay --
   fundamentals/earnings_revisions are mathematically inert in every backtest by design,
   since they only have a live snapshot; see extended_data_provider.py's lookahead-bias
   note), which ones have been helping vs. hurting since the backtest start date?

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

from sector_rotation_model import (
    RotationBacktester, DataProvider, MOMENTUM_COMPONENTS, TREND_COMPONENTS,
)


def _cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan


def _run_variant(provider: DataProvider, cfg: dict) -> dict:
    """Runs one backtest variant and extracts a compact stats row, including average
    monthly turnover and a portfolio-diversity check (not part of the core stats dict)
    as extra diagnostics."""
    bt = RotationBacktester(provider, cfg)
    result = bt.run()
    row = dict(result.stats)
    row["avg_monthly_turnover"] = float(result.turnover.mean()) if len(result.turnover) else np.nan
    row["n_rebalances"] = len(result.turnover)

    # Stability check: CAGR in each half of the sample. A factor that only helps in one half
    # is much more likely to be in-sample noise than a durable edge.
    eq = result.equity_curve.dropna()
    mid = len(eq) // 2
    row["cagr_h1"] = _cagr(eq.iloc[:mid + 1]) if mid > 1 else np.nan
    row["cagr_h2"] = _cagr(eq.iloc[mid:]) if mid > 1 else np.nan

    # Diagnostic: how many DISTINCT sector selections occurred across the whole
    # backtest. This catches degenerate/tied rankings -- e.g. a factor that's a
    # constant 0 for every sector produces a tie every month, which resolves to the
    # SAME fixed set of sectors every single rebalance (n_distinct_portfolios == 1).
    # A CAGR that looks good under those conditions is a static accidental bet on
    # whichever sectors won the tie-break, not a real signal -- this column makes
    # that visible instead of letting a good-looking but fake number pass unnoticed.
    sector_cols = [c for c in result.holdings_history.columns if c != "equity_exposure"]
    if sector_cols and len(result.holdings_history):
        held = result.holdings_history[sector_cols].fillna(0).gt(0)
        held_sets = held.apply(lambda r: tuple(sorted(r[r].index)), axis=1)
        row["n_distinct_portfolios"] = int(held_sets.nunique())
    else:
        row["n_distinct_portfolios"] = np.nan

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

    # The overbought overlay is not a weighted factor, so test it separately
    if base_cfg.get("overbought_enabled", True):
        variant_cfg = copy.deepcopy(base_cfg)
        variant_cfg["overbought_enabled"] = False
        rows["without overbought_overlay"] = _run_variant(provider, variant_cfg)

    df = pd.DataFrame(rows).T
    df["cagr_delta_if_removed"] = baseline["CAGR"] - df["CAGR"]
    df.loc["baseline (all factors)", "cagr_delta_if_removed"] = np.nan
    return _add_half_deltas(df, "baseline (all factors)")


def _add_half_deltas(df: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    """delta_h1 / delta_h2: baseline CAGR minus variant CAGR within each half of the sample.
    Same sign in both halves = consistent effect; opposite signs = likely noise."""
    base = df.loc[baseline_label]
    df["delta_h1"] = base["cagr_h1"] - df["cagr_h1"]
    df["delta_h2"] = base["cagr_h2"] - df["cagr_h2"]
    df.loc[baseline_label, ["delta_h1", "delta_h2"]] = np.nan
    return df


def run_component_attribution(provider: DataProvider, base_cfg: dict) -> pd.DataFrame:
    """Looks INSIDE the momentum and trend factors. Each is an average of several
    sub-signals (momentum: 12-1m, 6m, 3m, acceleration; trend: price vs 50d/200d MA, MA
    slope, relative strength, volume trend). Two views per component:
      - "without X": the full model with sub-signal X dropped from its factor
      - "only X":    the full model with X as the ONLY sub-signal in its factor
    Compare against the baseline row (current settings). Use `delta_h1`/`delta_h2` to see
    whether a component's effect is consistent across both halves of the sample.
    """
    rows = {"baseline (current settings)": _run_variant(provider, base_cfg)}
    for key, comps in (("momentum_components", MOMENTUM_COMPONENTS),
                       ("trend_components", TREND_COMPONENTS)):
        factor = key.split("_")[0]
        current = [c for c in base_cfg.get(key, comps)]
        for c in comps:
            if c in current and len(current) > 1:
                cfg = copy.deepcopy(base_cfg)
                cfg[key] = [x for x in current if x != c]
                rows[f"{factor}: without {c}"] = _run_variant(provider, cfg)
            cfg = copy.deepcopy(base_cfg)
            cfg[key] = [c]
            rows[f"{factor}: only {c}"] = _run_variant(provider, cfg)
    df = pd.DataFrame(rows).T
    base = df.loc["baseline (current settings)"]
    df["cagr_delta_vs_baseline"] = df["CAGR"] - base["CAGR"]
    df["delta_h1"] = df["cagr_h1"] - base["cagr_h1"]
    df["delta_h2"] = df["cagr_h2"] - base["cagr_h2"]
    df.loc["baseline (current settings)", ["cagr_delta_vs_baseline", "delta_h1", "delta_h2"]] = np.nan
    return df


def run_single_factor(provider: DataProvider, base_cfg: dict) -> pd.DataFrame:
    """For each factor, runs the model with ONLY that factor driving the ranking
    (weight 1.0, all others 0, overbought overlay off). Shows standalone performance -- useful alongside
    leave-one-out because a factor can look weak in combination but still be a
    genuinely strong standalone signal (or vice versa: a weak standalone factor can
    still add diversification value in the blend).

    CAVEAT: for fundamentals/earnings_revisions specifically (inert in every backtest --
    see module docstring), isolating them as the ONLY factor means every sector ties at a
    score of 0. The resulting "CAGR" is whatever fixed portfolio falls out of that tie --
    NOT a real standalone signal. Expect their rows to be identical to each other, and the
    n_distinct_portfolios column to read 1; the leave-one-out view already proves they
    contribute nothing either way.
    """
    rows = {}
    for factor in base_cfg["factor_weights"]:
        variant_cfg = copy.deepcopy(base_cfg)
        variant_cfg["factor_weights"] = {f: (1.0 if f == factor else 0.0) for f in base_cfg["factor_weights"]}
        # Isolate the factor: with the overbought overlay on, it would break the tie for
        # inert factors and make them look like a real RSI-driven rotation.
        variant_cfg["overbought_enabled"] = False
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


def format_component_report(comp_df: pd.DataFrame) -> str:
    cols = ["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_vs_baseline", "delta_h1", "delta_h2", "avg_monthly_turnover"]
    return "\n".join([
        "=" * 70,
        "COMPONENT ATTRIBUTION -- inside momentum and trend",
        "(delta_h1/delta_h2 = CAGR change vs baseline in each half of the sample;",
        " same sign in both halves = consistent, opposite signs = probably noise)",
        "=" * 70,
        comp_df[cols].to_string(),
    ])


def format_report(loo_df: pd.DataFrame, sf_df: pd.DataFrame, sweep_df: pd.DataFrame) -> str:
    """Plain-text summary for terminal/log output."""
    lines = []
    lines.append("=" * 70)
    lines.append("FACTOR ATTRIBUTION -- Leave-One-Out")
    lines.append("(positive cagr_delta_if_removed = factor is net ADDITIVE, keep it)")
    lines.append("(negative cagr_delta_if_removed = factor is a net DRAG, consider cutting)")
    lines.append("=" * 70)
    lines.append(loo_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_if_removed", "delta_h1", "delta_h2", "n_distinct_portfolios"]].to_string())
    lines.append("")
    lines.append("=" * 70)
    lines.append("FACTOR ATTRIBUTION -- Single-Factor Standalone")
    lines.append("=" * 70)
    sf_sorted = sf_df[["CAGR", "Sharpe", "MaxDrawdown", "n_distinct_portfolios"]].sort_values("CAGR", ascending=False)
    degenerate = sf_sorted[sf_sorted["n_distinct_portfolios"] <= 1]
    if not degenerate.empty:
        lines.append(f"*** WARNING: {', '.join(degenerate.index)} held a single static portfolio the ***")
        lines.append("*** entire backtest (n_distinct_portfolios=1) -- a tied/degenerate ranking,   ***")
        lines.append("*** not a real signal. Their CAGR is an accidental static bet, not evidence   ***")
        lines.append("*** those factors are working. Ignore these rows for weighting decisions.     ***")
        lines.append("")
    lines.append(sf_sorted.to_string())
    lines.append("")
    lines.append("=" * 70)
    lines.append("DIVERSIFICATION SENSITIVITY -- top_n x max_weight")
    lines.append("=" * 70)
    pivot = sweep_df.pivot(index="top_n", columns="max_weight", values="CAGR")
    lines.append(pivot.to_string())
    return "\n".join(lines)
