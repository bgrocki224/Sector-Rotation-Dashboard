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
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from sector_rotation_model import (
    SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG,
    DataProvider, FactorEngine, capture_ratios, CompositeScorer, PortfolioConstructor,
    RiskOverlay, RotationBacktester, zscore, apply_overbought_overlay,
)
from extended_data_provider import YFinanceExtendedProvider
from diagnostics import run_leave_one_out, run_single_factor, run_concentration_sweep, run_component_attribution

st.set_page_config(page_title="Sector Rotation Dashboard", layout="wide")

CACHE_TTL_SECONDS = 60 * 60 * 4  # re-pull prices at most every 4 hours
RETRY_AFTER_FAILURE_SECONDS = 60 * 5  # a live-data fetch with gaps is retried after 5 minutes, not 4 hours


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


@st.cache_resource(show_spinner=False)
def _live_score_store() -> dict:
    return {}


def _score_uncached(start_date: str, weights: dict, use_extended: bool, standardize: bool = False):
    # Extended provider is safe HERE: only today's snapshot is ever requested,
    # so there's no lookahead concern (see extended_data_provider.py warning).
    prices, volume, cached_provider = load_data(start_date, use_extended)
    # st.cache_data hands back an UNPICKLED copy of the provider, which can be an instance of an
    # older version of the class after a code update. Build a fresh provider for scoring and
    # reuse only the cached price data.
    provider_cls = YFinanceExtendedProvider if use_extended else DataProvider
    provider = provider_cls(tickers=list(cached_provider.tickers), start=start_date)
    provider._prices, provider._volume = prices, volume
    provider._macro, provider.macro_error = getattr(cached_provider, "_macro", None), getattr(cached_provider, "macro_error", None)
    tickers = [t for t in SECTOR_UNIVERSE if t in prices.columns]
    live_cfg = dict(CONFIG)
    live_cfg["standardize_components"] = standardize
    engine = FactorEngine(prices, volume, provider, live_cfg)
    as_of = prices.index[-1]
    factors = engine.all_factors(as_of, tickers)
    scorer = CompositeScorer(weights)
    scores = scorer.score(factors)
    scores = apply_overbought_overlay(scores, engine.rsi(as_of, tickers), CONFIG)
    health = provider.data_health() if hasattr(provider, "data_health") else None
    errors = list(getattr(provider, "fetch_errors", []))
    return factors, scores, as_of, health, errors


def compute_current_scores(start_date: str, weights: dict, use_extended: bool, standardize: bool = False):
    """Live scoring with a failure-aware cache. st.cache_data would have kept a throttled,
    all-NaN Yahoo result for 4 hours, so instead a result is cached for the full TTL only
    if fundamentals and earnings revisions both covered most sectors; otherwise it is
    retried after 5 minutes."""
    key = (start_date, tuple(sorted(weights.items())), use_extended, standardize)
    store = _live_score_store()
    hit = store.get(key)
    if hit:
        ttl = CACHE_TTL_SECONDS if hit["healthy"] else RETRY_AFTER_FAILURE_SECONDS
        if time.time() - hit["t"] < ttl:
            return hit["value"]
    with st.spinner("Scoring sectors (first load fetches live analyst data and can take a minute)..."):
        value = _score_uncached(start_date, weights, use_extended, standardize)
    factors = value[0]
    # Fundamentals decide cache health. Earnings revisions come only from Yahoo, which blocks
    # Streamlit Cloud, so requiring them would re-hit Yahoo every 5 minutes for nothing.
    healthy = (not use_extended) or bool(factors["fundamentals"].notna().sum() >= 8)
    store[key] = {"t": time.time(), "healthy": bool(healthy), "value": value}
    return value


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
    # fundamentals/earnings_revisions on the extended provider only have today's
    # snapshot, so using them across historical rebalance dates would be lookahead
    # bias. Those two are neutral (0) in this backtest; momentum, trend, seasonality,
    # macro_regime (real FRED history) and the RSI overbought overlay (prices) all
    # drive it. See extended_data_provider.py.
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
    "Use live fundamentals/earnings-revision proxies",
    value=True,
    help="When on, fundamentals (State Street's forward P/E per sector ETF) and earnings_revisions "
         "(net analyst EPS-revision breadth) come from free data. When off, those "
         "two are neutral (0) and the ranking runs on momentum/trend/seasonality/"
         "macro_regime plus the overbought overlay. Turn off if you hit Yahoo rate limits.",
)

st.sidebar.subheader("Factor weights")
weights = {}
for factor, default_w in CONFIG["factor_weights"].items():
    weights[factor] = st.sidebar.slider(factor.replace("_", " ").title(), 0.0, 1.0, float(default_w), 0.05)

standardize = st.sidebar.checkbox(
    "Standardize momentum/trend components", value=bool(CONFIG.get("standardize_components", False)),
    help="Momentum and trend are each an average of several sub-signals measured in different "
         "units (e.g. a 12-month return vs a volume-surge ratio). By default the raw numbers "
         "are averaged, so the sub-signal with the biggest range dominates. Turning this on "
         "z-scores each sub-signal across sectors first so they count equally. Use the "
         "Component Attribution diagnostic to compare.",
)

cfg = dict(CONFIG)
cfg["factor_weights"] = weights
cfg["standardize_components"] = standardize
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

factors, scores, as_of, live_health, live_errors = compute_current_scores(start_date, weights, use_extended, standardize)
next_rebal = next_rebalance_date(as_of)
days_to_rebal = (next_rebal - as_of).days

if get_base_provider(start_date).load_macro_data() is None:
    st.warning(
        "⚠️ FRED macro data could not be loaded, so `macro_regime` (the highest-weighted "
        "factor) is running NEUTRAL for every sector, in the live ranking AND the backtest. "
        "Results below are effectively the model without its macro factor. Check the app's "
        "network access to fred.stlouisfed.org, then use 'Force refresh' in the sidebar. "
        f"Detail: {get_base_provider(start_date).macro_error}"
    )

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
st.plotly_chart(fig_rank, width="stretch")

# --- Target portfolio ---
st.subheader("Target Portfolio (as of last close)")
constructor = PortfolioConstructor(cfg)
target_weights = constructor.target_weights(scores, current_holdings=set())
if target_weights:
    tw_df = pd.Series(target_weights, name="Weight").sort_values(ascending=False).to_frame()
    tw_df["Sector"] = [SECTOR_UNIVERSE.get(t, t) for t in tw_df.index]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(tw_df[["Sector", "Weight"]].style.format({"Weight": "{:.1%}"}), width="stretch")
    with c2:
        fig_pie = go.Figure(go.Pie(labels=tw_df["Sector"], values=tw_df["Weight"], hole=0.4))
        fig_pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, width="stretch")
else:
    st.warning("No sectors currently qualify for inclusion under the current thresholds.")

# --- Factor breakdown table ---
st.subheader("Raw Factor Values by Sector")
display_factors = factors.copy()
display_factors.index = [f"{t} ({SECTOR_UNIVERSE.get(t, t)})" for t in display_factors.index]
st.dataframe(display_factors.style.background_gradient(cmap="RdYlGn", axis=0).format("{:.3f}"),
             width="stretch")

# --- Live data health: why fundamentals / earnings_revisions might be flat ---
st.caption(
    "A flat or blank `fundamentals` / `earnings_revisions` column means that factor is NEUTRAL "
    "(contributes 0). That is by design in the backtest and diagnostics (no historical data), "
    "but in this live table it means the fetch failed or the proxies are switched off."
)
if not use_extended:
    st.info("Live proxies are switched off in the sidebar, so fundamentals and earnings_revisions are neutral (0).")
else:
    n_sec = len(factors)
    cov_f = int(factors["fundamentals"].notna().sum())
    cov_r = int(factors["earnings_revisions"].notna().sum())
    if cov_f < n_sec or cov_r < n_sec:
        st.warning(
            f"⚠️ Live data gaps: fundamentals available for {cov_f}/{n_sec} sectors, earnings "
            f"revisions for {cov_r}/{n_sec}. Missing sectors are running NEUTRAL (0) for that "
            "factor. Open 'Live data health' below for the reason. A failed fetch is retried "
            "after 5 minutes (or press 'Force refresh')."
        )
    with st.expander("Live data health (fundamentals & earnings revisions)"):
        if live_health is not None:
            st.dataframe(live_health, width="stretch")
            st.caption(
                "holdings_source = where each ETF's top holdings came from ('static fallback' means "
                "Yahoo's holdings lookup failed and an approximate list of sector leaders was used "
                "instead). with_pe / with_revisions = how many holdings returned usable data."
            )
        import extended_data_provider as _edp
        st.caption(f"Loaded code: extended_data_provider has SPDR support = "
                   f"{hasattr(_edp, 'spdr_data')}; fundamentals source per sector is in `pe_source` above.")
        if st.button("Test State Street (SPDR) fetch now", key="test_spdr"):
            import spdr_data
            rows = {}
            for _t in SECTOR_UNIVERSE:
                _pe, _asof, _err = spdr_data.fetch_forward_pe(_t)
                rows[_t] = {"forward_pe": _pe, "as_of": _asof, "error": _err}
            st.dataframe(pd.DataFrame(rows).T, width="stretch")
        if live_errors:
            st.markdown("**Most recent fetch errors**")
            st.code("\n".join(live_errors[:15]))
        else:
            st.caption("No fetch errors recorded.")

st.divider()

# --- Backtest ---
st.subheader("Backtest: Strategy vs. SPY")
st.caption(
    "Backtest runs on momentum, trend, seasonality, macro_regime (FRED history) and the "
    "RSI overbought overlay, regardless of the sidebar toggle. fundamentals and "
    "earnings_revisions only exist as today's snapshot, so they stay neutral here to "
    "avoid lookahead bias. If FRED is unreachable, macro_regime silently goes neutral."
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
st.plotly_chart(fig_ec, width="stretch")

stat_cols = st.columns(4)
labels = [("CAGR", "CAGR"), ("Sharpe", "Sharpe"), ("MaxDrawdown", "Max Drawdown"), ("Volatility", "Volatility")]
for col, (key, label) in zip(stat_cols, labels):
    val = result.stats.get(key, np.nan)
    col.metric(label, f"{val:.1%}" if key != "Sharpe" else f"{val:.2f}")

# --- Upside / downside capture vs SPY ---
st.markdown("**Capture vs SPY** — target: CAGR within ~1% of SPY and downside capture of 60–70% or lower")
spy_full = prices[BENCHMARK].reindex(ec.index).ffill()
mid = len(ec) // 2
periods = {"Full period": (ec, spy_full), "First half": (ec.iloc[:mid + 1], spy_full.iloc[:mid + 1]),
           "Second half": (ec.iloc[mid:], spy_full.iloc[mid:])}
cap_rows = {}
for name, (e, b) in periods.items():
    if len(e) < 60:
        continue
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cg = (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1
    bg = (b.iloc[-1] / b.iloc[0]) ** (1 / yrs) - 1
    cap = capture_ratios(e / e.iloc[0], b / b.iloc[0])
    cap_rows[name] = {"Strategy CAGR": cg, "SPY CAGR": bg, "CAGR gap": cg - bg,
                      "Upside capture": cap["up_capture"], "Downside capture": cap["down_capture"],
                      "Strategy max DD": (e / e.cummax() - 1).min(), "SPY max DD": (b / b.cummax() - 1).min()}
cap_df = pd.DataFrame(cap_rows).T
st.dataframe(cap_df.style.format({
    "Strategy CAGR": "{:.1%}", "SPY CAGR": "{:.1%}", "CAGR gap": "{:+.1%}",
    "Upside capture": "{:.0%}", "Downside capture": "{:.0%}",
    "Strategy max DD": "{:.1%}", "SPY max DD": "{:.1%}"}), width="stretch")
if "Full period" in cap_df.index:
    r = cap_df.loc["Full period"]
    ok_gap, ok_down = r["CAGR gap"] >= -0.01, r["Downside capture"] <= 0.70
    st.caption(f"Goal check: CAGR within 1% of SPY — {'✅' if ok_gap else '❌'} ({r['CAGR gap']:+.1%}); "
               f"downside capture ≤ 70% — {'✅' if ok_down else '❌'} ({r['Downside capture']:.0%}). "
               "Capture ratios use monthly returns, so they're approximate; check both halves are "
               "consistent before trusting a full-period number.")

st.subheader("Rebalance History")
st.dataframe(result.holdings_history.tail(24).style.format("{:.1%}", na_rep=""), width="stretch")

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

diag_tab1, diag_tab2, diag_tab3 = st.tabs(["Factor Attribution", "Diversification Sensitivity", "Component Attribution"])

with diag_tab1:
    st.caption(
        "`fundamentals` / `earnings_revisions` only have today's live snapshot (see the "
        "note above), so they're mathematically inert in every backtest -- expect their "
        "rows below to show ~0 effect. `macro_regime` uses real FRED history, so it is "
        "tested here. That's not a bug, it's the proof."
    )
    if st.button("Run factor attribution", key="run_attr"):
        with st.spinner("Running leave-one-out and single-factor backtests..."):
            loo_df = run_leave_one_out(base_provider, cfg)
            sf_df = run_single_factor(base_provider, cfg)

        st.markdown("**Leave-one-out** — positive `cagr_delta_if_removed` = factor is net "
                    "additive (removing it hurt); negative = factor is a net drag (removing it helped).")
        loo_display = loo_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_if_removed", "up_capture", "down_capture", "delta_h1", "delta_h2", "n_distinct_portfolios"]].copy()
        st.dataframe(loo_display.style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "cagr_delta_if_removed": "{:+.2%}", "up_capture": "{:.0%}", "down_capture": "{:.0%}", "delta_h1": "{:+.2%}", "delta_h2": "{:+.2%}",
            "n_distinct_portfolios": "{:.0f}",
        }), width="stretch")

        fig_loo = go.Figure(go.Bar(
            x=loo_df["cagr_delta_if_removed"].drop("baseline (all factors)"),
            y=[i.replace("without ", "") for i in loo_df.index if i != "baseline (all factors)"],
            orientation="h",
            marker_color=["#2ca02c" if v > 0 else "#d62728"
                          for v in loo_df["cagr_delta_if_removed"].drop("baseline (all factors)")],
        ))
        fig_loo.update_layout(height=320, xaxis_title="CAGR delta if factor removed",
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_loo, width="stretch")

        st.markdown("**Single-factor standalone** — each factor driving the ranking alone.")
        sf_display = sf_df[["CAGR", "Sharpe", "MaxDrawdown", "n_distinct_portfolios"]].sort_values("CAGR", ascending=False)
        degenerate = sf_display[sf_display["n_distinct_portfolios"] <= 1]
        if not degenerate.empty:
            st.warning(
                f"⚠️ {', '.join(degenerate.index)} held the exact same static portfolio for "
                "every single rebalance (n_distinct_portfolios = 1) — that's the signature of a "
                "tied/degenerate ranking (a constant-zero factor), not a real signal. Their CAGR "
                "here is an accidental static bet on whichever sectors won the tie-break, "
                "regardless of how good the number looks. Ignore these rows for weighting decisions."
            )
        st.dataframe(sf_display.style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "n_distinct_portfolios": "{:.0f}",
        }), width="stretch")

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
        st.plotly_chart(fig_heat, width="stretch")

        st.dataframe(
            sweep_df[["top_n", "max_weight", "CAGR", "Sharpe", "MaxDrawdown", "avg_monthly_turnover"]]
            .style.format({"CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}", "avg_monthly_turnover": "{:.1%}"}),
            width="stretch",
        )

with diag_tab3:
    st.caption(
        "Looks inside momentum and trend. 'without X' drops one sub-signal from its factor; "
        "'only X' uses that sub-signal alone. `delta_h1` / `delta_h2` are the CAGR change vs "
        "the baseline in the first / second half of the sample — a real edge should have the "
        "same sign in both; opposite signs usually mean noise. Respects the standardize "
        "checkbox in the sidebar."
    )
    if st.button("Run component attribution", key="run_comp"):
        with st.spinner("Running component backtests..."):
            comp_df = run_component_attribution(base_provider, cfg)
        comp_display = comp_df[["CAGR", "Sharpe", "MaxDrawdown", "cagr_delta_vs_baseline",
                                 "delta_h1", "delta_h2", "avg_monthly_turnover"]]
        st.dataframe(comp_display.style.format({
            "CAGR": "{:.2%}", "Sharpe": "{:.2f}", "MaxDrawdown": "{:.2%}",
            "cagr_delta_vs_baseline": "{:+.2%}", "delta_h1": "{:+.2%}", "delta_h2": "{:+.2%}",
            "avg_monthly_turnover": "{:.1%}",
        }), width="stretch")

st.caption(
    "Research/engineering tool, not investment advice. Fundamentals and earnings revisions "
    "use free yfinance proxies for LIVE scoring above (see extended_data_provider.py) but "
    "are neutral (0) inside every backtest and the diagnostics by design, since they only "
    "have today's snapshot and using them historically would be lookahead bias. "
    "Macro regime and the overbought overlay are backtested on real history — see README.md."
)
