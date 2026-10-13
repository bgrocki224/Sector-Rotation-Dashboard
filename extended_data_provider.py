"""
Extended Data Provider — free yfinance-only proxies
======================================================
The base `DataProvider` in sector_rotation_model.py returns None (a neutral 0 factor) for
fundamentals and earnings revisions, because there is no free historical feed for them.

This provider fills both in for LIVE scoring using only yfinance (no API key):
  - fundamentals: average trailing P/E of each sector ETF's top holdings (down-weighted
    to 5% in CONFIG: static valuation level has weak evidence for sector rotation)
  - earnings_revisions: net analyst EPS-revision breadth across top holdings, from
    yfinance's eps_revisions table (up vs. down revisions over the last 30 days). This
    REPLACES the old static recommendation-level factor, which is largely priced in.

Macro regime and the RSI overbought overlay are NOT here: both use real historical
data (FRED, prices), so they live in sector_rotation_model.py and run in backtests too.

*** LOOKAHEAD BIAS WARNING FOR BACKTESTING ***
`get_fundamentals` and `get_earnings_revisions` only see yfinance's CURRENT snapshot and
ignore `as_of`. Fine for live scoring; in a historical backtest they would feed today's
data into decisions dated years ago. Use this provider for LIVE/CURRENT scoring only and
keep backtests on the base `DataProvider`. streamlit_app.py and check_rebalance_signal.py
are already wired that way.

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
        self._info_cache: dict[str, tuple[pd.Timestamp, dict]] = {}
        self.n_holdings = n_holdings

    # ------------------------------------------------------------------
    # Fundamentals: holdings-based average trailing P/E, cross-sector relative valuation
    # ------------------------------------------------------------------
    def get_fundamentals(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Proxy: average trailing P/E across each sector ETF's top holdings.

        NOTE: earlier versions of this method read the SECTOR ETF's own `.info`
        for 'trailingPE' -- Yahoo/yfinance frequently does not populate that field
        for fund/ETF tickers (a known gap, not sector-specific), which produced
        blank/NaN values across the board. This version instead averages the
        `trailingPE` of each ETF's top holdings (individual large-cap stocks
        reliably have this field populated), which is far more consistently
        available. It reuses the same cached holdings list as `get_earnings_revisions`
        and a shared per-ticker info cache, so this doesn't double the network calls.

        Sign convention: returns -1 * average P/E, so a CHEAPER sector produces a
        HIGHER raw value -- consistent with 'higher score = more attractive'
        everywhere else in the model.

        Fallback order if holdings-based P/E is unavailable: (1) the ETF ticker's
        own `.info['trailingPE']`, in case Yahoo does have it for that fund, then
        (2) `funds_data.equity_holdings` fund-level aggregate stats (schema can
        vary by yfinance version, probed defensively). If every path fails, that
        sector is NaN (neutral / 0 after z-scoring) rather than raising.
        """
        vals = {}
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            pe_samples = []

            # Primary: average P/E of top holdings
            for h in self._get_top_holdings(t):
                info = self._get_info(h)
                pe = info.get("trailingPE") if info else None
                if pe and pe > 0:
                    pe_samples.append(pe)

            # Fallback 1: the ETF ticker's own trailingPE, if present
            if not pe_samples:
                info = self._get_info(t)
                pe = info.get("trailingPE") if info else None
                if pe and pe > 0:
                    pe_samples.append(pe)

            # Fallback 2: fund-level aggregate stats via funds_data.equity_holdings
            if not pe_samples:
                pe = self._get_fund_aggregate_pe(t)
                if pe and pe > 0:
                    pe_samples.append(pe)

            vals[t] = -float(np.mean(pe_samples)) if pe_samples else np.nan
        return pd.Series(vals, name="fundamentals")

    # ------------------------------------------------------------------
    # Earnings revisions: net analyst EPS-revision breadth across top holdings
    # ------------------------------------------------------------------
    def get_earnings_revisions(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Proxy: for each top holding, net breadth = (up - down) / (up + down) over the
        last 30 days, read from yfinance's `eps_revisions` table and averaged across the
        forward periods it lists (current/next quarter and year). Averaged over each
        sector ETF's top holdings. Higher = more analysts raising estimates.

        Limitations: top ~8 holdings only, a 30-day window only, and yfinance's table
        schema has changed between versions, so column names are probed defensively; a
        holding with no usable data is skipped, and a sector with none comes back NaN
        (neutral after z-scoring). A paid feed (I/B/E/S, FactSet, Financial Modeling
        Prep) would give cleaner, full-constituent revision data.
        """
        import yfinance as yf
        vals = {}
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            breadths = []
            for h in self._get_top_holdings(t):
                b = self._holding_revision_breadth(yf, h)
                if b is not None:
                    breadths.append(b)
            vals[t] = float(np.mean(breadths)) if breadths else np.nan
        return pd.Series(vals, name="earnings_revisions")

    @staticmethod
    def _holding_revision_breadth(yf, ticker: str) -> Optional[float]:
        try:
            tbl = yf.Ticker(ticker).eps_revisions
        except Exception:
            return None
        if tbl is None or len(tbl) == 0:
            return None
        cols = {str(c).lower(): c for c in tbl.columns}
        up_col, down_col = cols.get("uplast30days"), cols.get("downlast30days")
        if up_col is None or down_col is None:
            return None
        per_period = []
        for _, row in tbl.iterrows():
            up, down = row[up_col], row[down_col]
            if pd.notna(up) and pd.notna(down) and (up + down) > 0:
                per_period.append((up - down) / (up + down))
        return float(np.mean(per_period)) if per_period else None

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

    # ------------------------------------------------------------------
    # Helper: cached per-ticker `.info` lookup, shared by fundamentals and
    # other factors so a holding's info is only fetched once per cache window,
    # not once per factor.
    # ------------------------------------------------------------------
    def _get_info(self, ticker: str, ttl_days: int = 1) -> dict:
        now = pd.Timestamp.now()
        cached = self._info_cache.get(ticker)
        if cached and (now - cached[0]) < pd.Timedelta(days=ttl_days):
            return cached[1]
        import yfinance as yf
        info: dict = {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        self._info_cache[ticker] = (now, info)
        return info

    # ------------------------------------------------------------------
    # Helper: fund-level aggregate P/E as a last-resort fallback for fundamentals.
    # Schema of `funds_data.equity_holdings` has varied across yfinance versions,
    # so this probes a couple of plausible shapes rather than assuming one.
    # ------------------------------------------------------------------
    def _get_fund_aggregate_pe(self, ticker: str) -> Optional[float]:
        import yfinance as yf
        try:
            eq = yf.Ticker(ticker).funds_data.equity_holdings
        except Exception:
            return None
        if eq is None:
            return None
        for key in ("priceToEarnings", "price_to_earnings", "peRatio"):
            try:
                if hasattr(eq, "get"):
                    val = eq.get(key)
                    if val is not None:
                        return float(val)
                if hasattr(eq, "index") and key in eq.index:
                    row = eq.loc[key]
                    val = row.iloc[0] if hasattr(row, "iloc") else row
                    if val is not None:
                        return float(val)
            except Exception:
                continue
        return None
