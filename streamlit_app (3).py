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

st.caption(
    "Research/engineering tool, not investment advice. Fundamentals, analyst sentiment, and "
    "market sentiment factors run in neutral mode until a real data feed is wired into "
    "DataProvider (see README.md)."
)
