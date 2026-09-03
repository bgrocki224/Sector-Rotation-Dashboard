"""
Sector Rotation — Live Dashboard
==================================
A Streamlit app to actively watch the sector rotation model: current factor
scores, target vs. actual portfolio weights, days to next rebalance, and a
running backtest equity curve vs. SPY.

Run locally:
    pip install streamlit yfinance pandas numpy plotly --break-system-packages
    streamlit run streamlit_app.py

Deploy so it's reachable from a browser anywhere (free tier available):
    1. Push this folder to a GitHub repo (include sector_rotation_model.py + this file
       + a requirements.txt: streamlit, yfinance, pandas, numpy, plotly)
    2. Go to share.streamlit.io, connect the repo, point it at streamlit_app.py
    3. It redeploys automatically on every git push, and can be set to re-run on a
       schedule via Streamlit's built-in caching TTL (see CACHE_TTL_SECONDS below) --
       every visit within the TTL window reuses cached data, every visit after that
       window re-pulls fresh prices, so the page is always "live enough" without
       hitting rate limits.

This file assumes sector_rotation_model.py is in the same directory / importable.
"""

import datetime as dt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sector_rotation_model import (
    SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG,
    DataProvider, FactorEngine, CompositeScorer, PortfolioConstructor,
    RiskOverlay, RotationBacktester, zscore,
)
from extended_data_provider import YFinanceExtendedProvider
from diagnostics import run_leave_one_out, run_single_factor, run_concentration_sweep

st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")

CACHE_TTL_SECONDS = 60 * 60 * 4  # re-pull prices at most every 4 hours


# ---------------------------------------------------------------------------
# Data loading (cached so the app doesn't hammer the data source on every click)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Loading prices...")
def load_data(start_date: str, use_extended: bool):
    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider_cls = YFinanceExtendedProvider if use_extended else DataProvider
    provider = provider_cls(tickers=tickers, start=start_date)
    prices = provider.load_prices()
    volume = provider.load_volume()
    return prices, volume, provider


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Scoring sectors...")
def compute_current_scores(start_date: str, weights: dict, use_extended: bool):
    # Extended provider is safe HERE: only today's snapshot is ever requested,
    # so there's no lookahead concern (see extended_data_provider.py warning).
    prices, volume, provider = load_data(start_date, use_extended)
    tickers = [t for t in SECTOR_UNIVERSE if t in prices.columns]
    engine = FactorEngine(prices, volume, provider)
    as_of = prices.index[-1]
    factors = engine.all_factors(as_of, tickers)
    scorer = CompositeScorer(weights)
    scores = scorer.score(factors)
    return factors, scores, as_of


@st.cache_resource(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_base_provider(start_date: str) -> DataProvider:
    """A warmed-up base DataProvider (price data pre-loaded), reused by the
    diagnostics tools below so every backtest variant they run shares the same cached
    prices instead of re-downloading. Uses st.cache_resource (not st.cache_data) since
    DataProvider is a stateful object, not plain serializable data."""
    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider = DataProvider(tickers=tickers, start=start_date)
    provider.load_prices()
    provider.load_volume()
    return provider


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Running backtest...")
def run_backtest(start_date: str, cfg: dict):
    # Intentionally ALWAYS uses the base DataProvider, never the extended one --
    # get_fundamentals/get_analyst_sentiment on the extended provider only have
    # today's snapshot, so using them across historical rebalance dates would
    # introduce lookahead bias. Fundamentals/analyst_sentiment/market_sentiment
    # are neutral (0) in this backtest; only momentum/trend/seasonality
    # (all correctly historical) drive it. See extended_data_provider.py.
    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider = DataProvider(tickers=tickers, start=start_date)
    bt = RotationBacktester(provider, cfg)
    return bt.run()


def next_rebalance_date(as_of: pd.Timestamp) -> pd.Timestamp:
    return (as_of + pd.offsets.MonthEnd(0)) if as_of.day < 28 else (as_of + pd.offsets.MonthEnd(1))


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Sector Rotation Model")
start_date = st.sidebar.date_input("Backtest start", dt.date(2015, 1, 1)).isoformat()

use_extended = st.sidebar.toggle(
    "Use live fundamentals/analyst/sentiment proxies",
    value=True,
    help="When on, fundamentals/analyst_sentiment/market_sentiment are computed from "
         "free yfinance data (ETF trailing P/E, holdings-weighted analyst ratings, "
         "RSI-based sentiment). When off, those three factors are neutral (0) and "
         "the ranking runs on momentum/trend/seasonality only. Turn off if "
         "you're hitting Yahoo Finance rate limits.",
)

st.sidebar.subheader("Factor weights")
weights = {}
for factor, default_w in CONFIG["factor_weights"].items():
    weights[factor] = st.sidebar.slider(factor.replace("_", " ").title(), 0.0, 1.0, float(default_w), 0.05)

cfg = dict(CONFIG)
cfg["factor_weights"] = weights
cfg["top_n"] = st.sidebar.slider("Top N sectors held", CONFIG["min_sectors_held"], 8, CONFIG["top_n"])
st.sidebar.caption(f"Portfolio never holds fewer than {CONFIG['min_sectors_held']} sectors at rebalance time.")

if st.sidebar.button("Force refresh (bypass cache)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(f"Data cached for up to {CACHE_TTL_SECONDS // 3600}h between pulls.")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("📊 Sector Rotation — Live Dashboard")

factors, scores, as_of = compute_current_scores(start_date, weights, use_extended)
next_rebal = next_rebalance_date(as_of)
days_to_rebal = (next_rebal - as_of).days

col1, col2, col3 = st.columns(3)
col1.metric("As of", as_of.strftime("%Y-%m-%d"))
col2.metric("Next rebalance", next_rebal.strftime("%Y-%m-%d"), f"{days_to_rebal}d away")
col3.metric("Sectors in universe", len(SECTOR_UNIVERSE))

st.divider()

# --- Current rankings ---
st.subheader("Current Sector Rankings")
rank_df = scores.rename("Composite Score").to_frame()
rank_df["Sector"] = [SECTOR_UNIVERSE.get(t, t) for t in rank_df.index]
rank_df = rank_df[["Sector", "Composite Score"]].sort_values("Composite Score", ascending=False)

fig_rank = go.Figure(go.Bar(
    x=rank_df["Composite Score"], y=rank_df.index, orientation="h",
    marker_color=["#2ca02c" if v > 0 else "#d62728" for v in rank_df["Composite Score"]],
))
fig_rank.update_layout(yaxis=dict(autorange="reversed"), height=400,
                        xaxis_title="Composite Z-Score", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_rank, use_container_width=True)

# --- Target portfolio ---
st.subheader("Target Portfolio (as of last close)")
constructor = PortfolioConstructor(cfg)
target_weights = constructor.target_weights(scores, current_holdings=set())
if target_weights:
    tw_df = pd.Series(target_weights, name="Weight").sort_values(ascending=False).to_frame()
    tw_df["Sector"] = [SECTOR_UNIVERSE.get(t, t) for t in tw_df.index]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(tw_df[["Sector", "Weight"]].style.format({"Weight": "{:.1%}"}), use_container_width=True)
    with c2:
        fig_pie = go.Figure(go.Pie(labels=tw_df["Sector"], values=tw_df["Weight"], hole=0.4))
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning("No sectors currently qualify for inclusion under the current thresholds.")

# --- Factor breakdown table ---
st.subheader("Raw Factor Values by Sector")
display_factors = factors.copy()
display_factors.index = [f"{t} ({SECTOR_UNIVERSE.get(t, t)})" for t in display_factors.index]
st.dataframe(display_factors.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.3f}"),
             use_container_width=True)

st.divider()

# --- Backtest ---
st.subheader("Backtest: Strategy vs. SPY")
st.caption(
    "Backtest always runs on momentum/trend/seasonality only, regardless of the "
    "sidebar toggle. Live fundamentals/analyst/sentiment data only exists as today's "
    "snapshot, so including it historically would introduce lookahead bias."
)
result = run_backtest(start_date, cfg)
ec = result.equity_curve
prices, _, _ = load_data(start_date, use_extended=False)
bench = prices[BENCHMARK].reindex(ec.index).ffill()
bench = bench / bench.iloc[0]

fig_ec = go.Figure()
fig_ec.add_trace(go.Scatter(x=ec.index, y=ec.values, name="Strategy", line=dict(color="#1f77b4", width=2)))
fig_ec.add_trace(go.Scatter(x=bench.index, y=bench.values, name="SPY (buy & hold)", line=dict(color="#888", width=1.5, dash="dot")))
fig_ec.update_layout(height=420, yaxis_title="Growth of $1", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig_ec, use_container_width=True)

stat_cols = st.columns(4)
labels = [("CAGR", "CAGR"), ("Sharpe", "Sharpe"), ("MaxDrawdown", "Max Drawdown"), ("Volatility", "Volatility")]
for col, (key, label) in zip(stat_cols, labels):
    val = result.stats.get(key, np.nan)
    col.metric(label, f"{val:.1%}" if key != "Sharpe" else f"{val:.2f}")

st.subheader("Rebalance History")
st.dataframe(result.holdings_history.tail(24).style.format("{:.1%}", na_rep=""), use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Diagnostics: factor attribution + diversification sensitivity
# ---------------------------------------------------------------------------

st.subheader("🔬 Diagnostics")
st.caption(
    "Both tools below run several full backtests (CPU-only, no extra network calls -- "
    "they reuse the price data already loaded above) to answer empirically, rather than "
    "by guessing, which factors and which diversification settings are actually helping "
    "or hurting since the backtest start date."
)

base_provider = get_base_provider(start_date)

diag_tab1, diag_tab2 = st.tabs(["Factor Attribution", "Diversification Sensitivity"])

with diag_tab1:
    st.caption(
        "`fundamentals` / `analyst_sentiment` / `market_sentiment` only have today's live "
        "snapshot (see the note above), so they're mathematically inert in every backtest -- "
        "expect their rows below to show ~0 effect. That's not a bug, it's the proof."
    )
    if st.button("Run factor attribution", key="run_attr"):
        with st.spinner("Running leave-one-out and single-factor backtests..."):
            loo_df = run_leave_one_out(base_provider, cfg)
            sf_df = run_single_factor(base_provider, cfg)

        st.markdown("**Leave-one-out** — positive `cagr_delta_if_removed` = factor is net "
                    "additive (removing it hurt); negative = factor is a net drag (removing it helped).")
        loo_display = loo_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_if_removed"]].copy()
        st.dataframe(loo_display.style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "cagr_delta_if_removed": "{:+.2%}",
        }), use_container_width=True)

        fig_loo = go.Figure(go.Bar(
            x=loo_df["cagr_delta_if_removed"].drop("baseline (all factors)"),
            y=[i.replace("without ", "") for i in loo_df.index if i != "baseline (all factors)"],
            orientation="h",
            marker_color=["#2ca02c" if v > 0 else "#d62728"
                          for v in loo_df["cagr_delta_if_removed"].drop("baseline (all factors)")],
        ))
        fig_loo.update_layout(height=320, xaxis_title="CAGR delta if factor removed",
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_loo, use_container_width=True)

        st.markdown("**Single-factor standalone** — each factor driving the ranking alone.")
        st.dataframe(sf_df[["CAGR", "Sharpe", "MaxDrawdown"]].sort_values("CAGR", ascending=False).style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}",
        }), use_container_width=True)

with diag_tab2:
    st.caption(
        "Sweeps `top_n` (sectors held) x `max_weight` (concentration cap) with "
        "`min_sectors_held` and `hysteresis_rank` set relative to `top_n` for each cell, "
        "so every combination is internally consistent."
    )
    sweep_top_n = st.multiselect("top_n values to test", [2, 3, 4, 5, 6, 7, 8], default=[2, 3, 4, 5, 6])
    sweep_max_w = st.multiselect("max_weight values to test", [0.25, 0.35, 0.50, 0.65, 1.00],
                                  default=[0.35, 0.50, 0.65, 1.00])
    if st.button("Run diversification sweep", key="run_sweep") and sweep_top_n and sweep_max_w:
        with st.spinner(f"Running {len(sweep_top_n) * len(sweep_max_w)} backtest variants..."):
            sweep_df = run_concentration_sweep(base_provider, cfg, sweep_top_n, sweep_max_w)

        pivot_cagr = sweep_df.pivot(index="top_n", columns="max_weight", values="CAGR")
        fig_heat = go.Figure(go.Heatmap(
            z=pivot_cagr.values, x=[f"{c:.0%}" for c in pivot_cagr.columns], y=pivot_cagr.index,
            colorscale="RdYlGn", text=[[f"{v:.1%}" for v in row] for row in pivot_cagr.values],
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(height=350, xaxis_title="max_weight", yaxis_title="top_n",
                                margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.dataframe(
            sweep_df[["top_n", "max_weight", "CAGR", "Sharpe", "MaxDrawdown", "avg_monthly_turnover"]]
            .style.format({"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "avg_monthly_turnover": "{:.1%}"}),
            use_container_width=True,
        )

st.caption(
    "Research/engineering tool, not investment advice. Fundamentals, analyst sentiment, and "
    "market sentiment factors run in neutral mode until a real data feed is wired into "
    "DataProvider (see README.md)."
)
