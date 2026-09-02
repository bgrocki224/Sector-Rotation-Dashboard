"""
Extended Data Provider — free yfinance-only proxies
======================================================
The base `DataProvider` in sector_rotation_model.py leaves fundamentals, analyst
sentiment, and market sentiment as neutral placeholders (they return None, which
z-scores to 0 for every sector, i.e. no effect on rankings) because there's no
universal free feed for ETF-level fundamentals or analyst data.

This provider fills those three in using only yfinance -- no paid API key needed --
by approximating each factor from data yfinance already exposes. These are PROXIES,
not the clean factor a paid data vendor (FMP, Zacks, Refinitiv) would give you.
Read each method's docstring for exactly what's being approximated and its
limitations before trusting it.

*** IMPORTANT: LOOKAHEAD BIAS WARNING FOR BACKTESTING ***
`get_fundamentals` and `get_analyst_sentiment` only have access to yfinance's CURRENT
snapshot (there's no free historical time series for ETF-level P/E or analyst
ratings) -- they IGNORE the `as_of` argument and always return today's values.
That's fine for live/current rankings, but using this provider inside a historical
BACKTEST would silently feed today's fundamentals into decisions dated years ago
-- a real lookahead bias that would make backtest results look better than a live
strategy could have actually achieved. `get_market_sentiment` (RSI, derived from
price history already loaded) does NOT have this problem and is safe historically.

For this reason: use `YFinanceExtendedProvider` for LIVE/CURRENT scoring only.
Keep backtests on the base `DataProvider` (momentum/trend/seasonality/calendar,
all of which are computed correctly as-of each historical date). The bundled
streamlit_app.py and check_rebalance_signal.py are already wired this way.

Usage:
    from extended_data_provider import YFinanceExtendedProvider
    provider = YFinanceExtendedProvider(tickers=tickers, start="2015-01-01")
    # use provider anywhere the base DataProvider was used

Performance/rate-limit note: fetching analyst data for each sector's top holdings
means ~8-10 extra network calls per sector (~90-110 total). Yahoo Finance rate-limits
by IP, which matters more on a shared host like Streamlit Community Cloud than
locally. Holdings composition is cached for 7 days by default since it rarely
changes; combine this with the dashboard's existing 4-hour data cache
(CACHE_TTL_SECONDS in streamlit_app.py) to keep total calls reasonable. If you hit
rate limits, raise CACHE_TTL_SECONDS, raise holdings_cache_ttl_days, or reduce
n_holdings below.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

from sector_rotation_model import DataProvider, SECTOR_UNIVERSE


class YFinanceExtendedProvider(DataProvider):

    def __init__(self, *args, holdings_cache_ttl_days: int = 7, n_holdings: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self._holdings_cache: dict[str, tuple[pd.Timestamp, list[str]]] = {}
        self._holdings_cache_ttl = pd.Timedelta(days=holdings_cache_ttl_days)
        self.n_holdings = n_holdings

    # ------------------------------------------------------------------
    # Fundamentals: ETF-level trailing P/E, cross-sector relative valuation
    # ------------------------------------------------------------------
    def get_fundamentals(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Proxy: each sector ETF's trailing P/E, as reported by Yahoo (Yahoo
        aggregates this from the underlying holdings for you -- one call per ETF,
        no need to fetch each holding individually).

        Sign convention: returns -1 * P/E, so a CHEAPER sector produces a HIGHER
        raw value -- consistent with 'higher score = more attractive' everywhere
        else in the model. This gets z-scored downstream like every other factor.

        Limitation: this is a valuation SNAPSHOT (cheap/expensive relative to its
        10 sector peers right now), not valuation relative to the sector's OWN
        history (e.g. "cheap for tech, expensive for tech historically") -- that
        needs a fundamentals time series, which requires a paid vendor. yfinance's
        `.info` can also be missing this field for some tickers; those come back
        as NaN and get treated as neutral (0) by the z-score step.
        """
        import yfinance as yf
        vals = {}
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            try:
                info = yf.Ticker(t).info
                pe = info.get("trailingPE")
                vals[t] = -pe if pe else np.nan
            except Exception:
                vals[t] = np.nan
        return pd.Series(vals, name="fundamentals")

    # ------------------------------------------------------------------
    # Analyst sentiment: holdings-weighted analyst recommendation
    # ------------------------------------------------------------------
    def get_analyst_sentiment(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Proxy: average sell-side `recommendationMean` (Yahoo's 1=Strong Buy ...
        5=Sell scale) across each sector ETF's top N holdings, sign-flipped so a
        HIGHER raw value means MORE bullish -- again consistent with the rest of
        the model's convention.

        Limitation: this reflects current analyst stance on the top ~8 holdings,
        not the full fund, and not revision MOMENTUM (upgrades/downgrades over
        time) which is what a real "analyst sentiment" factor usually means --
        that requires a paid estimates-revision feed. Individual-stock
        `recommendationMean` on Yahoo is also sometimes stale. Treat this as a
        rough analyst-stance proxy, not a precise revisions signal.
        """
        import yfinance as yf
        vals = {}
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            holdings = self._get_top_holdings(t)
            if not holdings:
                vals[t] = np.nan
                continue
            scores = []
            for h in holdings:
                try:
                    info = yf.Ticker(h).info
                    rec = info.get("recommendationMean")
                    if rec:
                        scores.append(-rec)  # flip: lower rec (Strong Buy=1) -> higher score
                except Exception:
                    continue
            vals[t] = float(np.mean(scores)) if scores else np.nan
        return pd.Series(vals, name="analyst_sentiment")

    # ------------------------------------------------------------------
    # Market sentiment: RSI-based positioning/stretch proxy
    # ------------------------------------------------------------------
    def get_market_sentiment(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Proxy: 14-day RSI, distance from the neutral 50 line, computed from price
        data already loaded (no extra network calls). Treats moderate strength
        (RSI 50-80) as a positive sentiment signal but penalizes extreme overbought
        readings (RSI > 80) as stretched/crowded rather than continuing to reward them.

        Limitation: this is a technical positioning proxy, not survey-based (AAII/
        NAAIM) or options-based (put/call ratio) sentiment, which is what the
        factor is meant to capture -- and it overlaps somewhat with the `trend`
        factor already in the model, so don't expect it to add fully independent
        information. A real options/survey feed would be a cleaner signal if you
        want to replace this later.
        """
        vals = {}
        hist = self.load_prices().loc[:as_of]
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            px = hist[t].dropna() if t in hist.columns else pd.Series(dtype=float)
            if len(px) < 20:
                vals[t] = np.nan
                continue
            delta = px.diff().dropna()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            r = rsi.iloc[-1]
            if pd.isna(r):
                vals[t] = np.nan
            elif r > 80:
                vals[t] = 50 - r  # extreme overbought -> penalize
            else:
                vals[t] = r - 50
        return pd.Series(vals, name="market_sentiment")

    # ------------------------------------------------------------------
    # Helper: cached top-holdings lookup
    # ------------------------------------------------------------------
    def _get_top_holdings(self, ticker: str) -> list[str]:
        now = pd.Timestamp.now()
        cached = self._holdings_cache.get(ticker)
        if cached and (now - cached[0]) < self._holdings_cache_ttl:
            return cached[1]
        import yfinance as yf
        holdings: list[str] = []
        try:
            fd = yf.Ticker(ticker).funds_data
            top = fd.top_holdings
            if top is not None and len(top) > 0:
                holdings = list(top.index[: self.n_holdings])
        except Exception:
            holdings = []
        self._holdings_cache[ticker] = (now, holdings)
        return holdings
