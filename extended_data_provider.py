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
import time
from typing import Optional

from sector_rotation_model import DataProvider, SECTOR_UNIVERSE
import spdr_data


# Approximate large-cap constituents per sector ETF, used ONLY if Yahoo's holdings endpoint
# (funds_data.top_holdings) fails or returns nothing. Both live factors depend on knowing the
# holdings, so without a fallback one blocked endpoint silently zeroes both of them. These are
# long-standing sector leaders, not an exact or current ETF weighting -- the data-health panel
# labels any sector running on this list so it is never mistaken for the real thing.
STATIC_FALLBACK_HOLDINGS = {
    "XLK":  ["NVDA", "AAPL", "MSFT", "AVGO", "ORCL", "CRM", "CSCO", "AMD"],
    "XLF":  ["BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS"],
    "XLE":  ["XOM", "CVX", "COP", "WMB", "EOG", "SLB", "KMI", "OKE"],
    "XLV":  ["LLY", "JNJ", "ABBV", "UNH", "MRK", "ABT", "TMO", "ISRG"],
    "XLI":  ["GE", "RTX", "CAT", "UBER", "BA", "HON", "UNP", "DE"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "BKNG", "TJX", "LOW", "SBUX"],
    "XLP":  ["COST", "WMT", "PG", "KO", "PM", "PEP", "MDLZ", "MO"],
    "XLU":  ["NEE", "SO", "DUK", "CEG", "AEP", "SRE", "D", "EXC"],
    "XLB":  ["LIN", "SHW", "ECL", "FCX", "APD", "NEM", "CTVA", "MLM"],
    "XLRE": ["PLD", "AMT", "WELL", "EQIX", "SPG", "DLR", "PSA", "O"],
    "XLC":  ["META", "GOOGL", "NFLX", "T", "VZ", "DIS", "CMCSA", "TMUS"],
}

_RATE_LIMIT_HINTS = ("429", "too many requests", "rate limit", "rate-limit", "ratelimit")


class YFinanceExtendedProvider(DataProvider):

    def __init__(self, *args, holdings_cache_ttl_days: int = 7, n_holdings: int = 8,
                 retry_wait_seconds: float = 1.5, use_spdr: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # cache entries: (timestamp, holdings, source) -- only REAL yfinance results get the
        # long TTL; fallback/failed lookups are retried after 15 minutes instead of sticking.
        self._holdings_cache: dict[str, tuple[pd.Timestamp, list[str], str]] = {}
        self._holdings_cache_ttl = pd.Timedelta(days=holdings_cache_ttl_days)
        self._info_cache: dict[str, tuple[pd.Timestamp, dict]] = {}  # successes only
        self.n_holdings = n_holdings
        self.retry_wait_seconds = retry_wait_seconds
        self.use_spdr = use_spdr
        self.spdr_as_of: dict[str, str] = {}       # sector -> 'as of' date on SPDR's page
        self.fetch_errors: list[str] = []          # human-readable failure reasons
        self._health: dict[str, dict] = {}         # per-sector coverage, see data_health()

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------
    def _record_error(self, step: str, ticker: str, err) -> None:
        msg = f"{step} [{ticker}]: {type(err).__name__}: {str(err)[:140]}" if isinstance(err, BaseException) \
            else f"{step} [{ticker}]: {err}"
        if msg not in self.fetch_errors and len(self.fetch_errors) < 40:
            self.fetch_errors.append(msg)

    def _call_with_retry(self, fn, step: str, ticker: str):
        """Runs fn(); on a rate-limit style error waits once and retries. Records every
        failure instead of swallowing it. Returns (ok, value)."""
        for attempt in (1, 2):
            try:
                return True, fn()
            except Exception as e:
                limited = any(h in str(e).lower() for h in _RATE_LIMIT_HINTS)
                if attempt == 1 and limited:
                    time.sleep(self.retry_wait_seconds)
                    continue
                self._record_error(step, ticker, e)
                return False, None
        return False, None

    def _set_health(self, sector: str, **kw) -> None:
        self._health.setdefault(sector, {}).update(kw)

    def data_health(self) -> pd.DataFrame:
        """Per-sector coverage of the live factors: where holdings came from, how many
        holdings returned a usable P/E and a usable revision reading. A sector with 0 in
        a column is running NEUTRAL (0) for that factor."""
        rows = {}
        for t in SECTOR_UNIVERSE:
            h = self._health.get(t, {})
            rows[t] = {
                "holdings_source": h.get("holdings_source", "not fetched"),
                "holdings": h.get("holdings", 0),
                "with_pe": h.get("with_pe", 0),
                "with_revisions": h.get("with_revisions", 0),
                "pe_source": h.get("pe_source", ""),
            }
        df = pd.DataFrame(rows).T
        for c in ("holdings", "with_pe", "with_revisions"):
            df[c] = df[c].astype(int)
        return df

    # ------------------------------------------------------------------
    # Fundamentals: holdings-based average trailing P/E
    # ------------------------------------------------------------------
    def get_fundamentals(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Average trailing P/E across each sector ETF's top holdings, sign-flipped so a
        CHEAPER sector scores HIGHER (consistent with 'higher = more attractive').

        Fallbacks if no holding returns a P/E: the ETF's own `.info['trailingPE']`, then
        the fund-level 'Price/Earnings' row of `funds_data.equity_holdings`. If every path
        fails the sector is NaN (neutral after z-scoring) AND the reason is recorded in
        `fetch_errors` / `data_health()`.

        Lookahead warning: current snapshot only -- live scoring, never backtests."""
        spdr = self._get_spdr_forward_pe() if self.use_spdr else {}
        if spdr:
            # One consistent source for every sector (forward P/E, issuer-published). Mixing it
            # with trailing P/Es from Yahoo for the leftovers would compare different measures,
            # so sectors it missed stay NaN (neutral) instead.
            vals = {}
            for t in self.tickers:
                if t not in SECTOR_UNIVERSE:
                    continue
                pe = spdr.get(t)
                self._set_health(t, with_pe=0, pe_source="spdr_forward_pe" if pe else "none")
                vals[t] = -float(pe) if pe else np.nan
            return pd.Series(vals, name="fundamentals")

        vals = {}
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            pe_samples, source = [], ""
            for h in self._get_top_holdings(t):
                pe = self._get_info(h).get("trailingPE")
                if pe and pe > 0:
                    pe_samples.append(pe)
            if pe_samples:
                source = "holdings"
            if not pe_samples:
                pe = self._get_info(t).get("trailingPE")
                if pe and pe > 0:
                    pe_samples.append(pe); source = "etf_info"
            if not pe_samples:
                pe = self._get_fund_aggregate_pe(t)
                if pe and pe > 0:
                    pe_samples.append(pe); source = "fund_aggregate"
            self._set_health(t, with_pe=len(pe_samples) if source == "holdings" else 0, pe_source=source or "none")
            vals[t] = -float(np.mean(pe_samples)) if pe_samples else np.nan
        return pd.Series(vals, name="fundamentals")

    def _get_spdr_forward_pe(self, min_coverage: int = 9) -> dict[str, float]:
        """Forward P/E for each sector ETF from State Street's own product pages (see
        spdr_data.py). Returns {} -- so the caller falls back to Yahoo -- unless at least
        `min_coverage` sectors come back. Successful results are cached for 6 hours."""
        cached = getattr(self, "_spdr_cache", None)
        if cached and time.time() - cached[0] < 6 * 3600:
            return cached[1]
        out, errs = {}, []
        for t in self.tickers:
            if t not in SECTOR_UNIVERSE:
                continue
            pe, as_of, err = spdr_data.fetch_forward_pe(t)
            if pe is not None:
                out[t] = pe
                if as_of:
                    self.spdr_as_of[t] = as_of
            else:
                errs.append((t, err))
        if len(out) < min_coverage:
            for t, err in errs:
                self._record_error("spdr_forward_pe", t, err)
            return {}
        for t, err in errs:
            self._record_error("spdr_forward_pe (sector left neutral)", t, err)
        stale = [d for d in (spdr_data.days_old(a) for a in set(self.spdr_as_of.values())) if d is not None and d > 10]
        if stale:
            self._record_error("spdr_forward_pe", "all", f"page data is {max(stale)} days old")
        self._spdr_cache = (time.time(), out)
        return out

    # ------------------------------------------------------------------
    # Earnings revisions: net analyst EPS-revision breadth across top holdings
    # ------------------------------------------------------------------
    def get_earnings_revisions(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """For each top holding, net breadth = (up - down) / (up + down) over the last 30
        days from yfinance's `eps_revisions` table, averaged over the forward periods it
        lists, then averaged over the sector's holdings. Higher = more analysts raising.

        Limitations: top ~8 holdings, 30-day window, and yfinance's table schema has changed
        between versions (column names probed defensively; unreadable holdings are skipped
        and the reason recorded). A paid feed (I/B/E/S, FactSet, Financial Modeling Prep)
        would give full-constituent data. Lookahead warning: snapshot only -- live use."""
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
            self._set_health(t, with_revisions=len(breadths))
            vals[t] = float(np.mean(breadths)) if breadths else np.nan
        return pd.Series(vals, name="earnings_revisions")

    def _holding_revision_breadth(self, yf, ticker: str) -> Optional[float]:
        ok, tbl = self._call_with_retry(lambda: yf.Ticker(ticker).eps_revisions, "eps_revisions", ticker)
        if not ok:
            return None
        if tbl is None or len(tbl) == 0:
            self._record_error("eps_revisions", ticker, "empty table returned")
            return None
        cols = {str(c).lower(): c for c in tbl.columns}
        up_col, down_col = cols.get("uplast30days"), cols.get("downlast30days")
        if up_col is None or down_col is None:
            self._record_error("eps_revisions", ticker, f"expected upLast30days/downLast30days, got {list(tbl.columns)}")
            return None
        per_period = []
        for _, row in tbl.iterrows():
            up, down = row[up_col], row[down_col]
            if pd.notna(up) and pd.notna(down) and (up + down) > 0:
                per_period.append((up - down) / (up + down))
        if not per_period:
            self._record_error("eps_revisions", ticker, "no analyst revisions in the last 30 days")
            return None
        return float(np.mean(per_period))

    # ------------------------------------------------------------------
    # Helpers: holdings (with static fallback), per-ticker info, fund-level P/E
    # ------------------------------------------------------------------
    def _get_top_holdings(self, ticker: str) -> list[str]:
        now = pd.Timestamp.now()
        cached = self._holdings_cache.get(ticker)
        if cached:
            ttl = self._holdings_cache_ttl if cached[2] == "yfinance" else pd.Timedelta(minutes=15)
            if (now - cached[0]) < ttl:
                return cached[1]

        import yfinance as yf
        holdings: list[str] = []
        source = "yfinance"

        def _fetch():
            top = yf.Ticker(ticker).funds_data.top_holdings
            return list(top.index[: self.n_holdings]) if top is not None and len(top) > 0 else []

        ok, got = self._call_with_retry(_fetch, "funds_data.top_holdings", ticker)
        if ok and got:
            holdings = [str(h) for h in got]
        else:
            if ok and not got:
                self._record_error("funds_data.top_holdings", ticker, "returned no holdings")
            holdings = list(STATIC_FALLBACK_HOLDINGS.get(ticker, []))[: self.n_holdings]
            source = "static fallback" if holdings else "none"

        self._holdings_cache[ticker] = (now, holdings, source)
        self._set_health(ticker, holdings_source=source, holdings=len(holdings))
        return holdings

    def _get_info(self, ticker: str, ttl_days: int = 1) -> dict:
        """Per-ticker `.info`, shared across factors. Only NON-EMPTY results are cached, so
        a throttled/failed call is retried next time instead of sticking for a day."""
        now = pd.Timestamp.now()
        cached = self._info_cache.get(ticker)
        if cached and (now - cached[0]) < pd.Timedelta(days=ttl_days):
            return cached[1]
        import yfinance as yf
        ok, info = self._call_with_retry(lambda: yf.Ticker(ticker).info, "info", ticker)
        info = (info or {}) if ok else {}
        if info:
            self._info_cache[ticker] = (now, info)
        elif ok:
            self._record_error("info", ticker, "empty info returned")
        return info

    def _get_fund_aggregate_pe(self, ticker: str) -> Optional[float]:
        """Last-resort fund-level P/E from `funds_data.equity_holdings`. yfinance returns a
        DataFrame indexed by labels such as 'Price/Earnings' (earlier versions of this code
        probed names that don't exist there); the first column is the fund itself."""
        import yfinance as yf
        ok, eq = self._call_with_retry(lambda: yf.Ticker(ticker).funds_data.equity_holdings, "equity_holdings", ticker)
        if not ok or eq is None:
            return None
        for key in ("Price/Earnings", "priceToEarnings", "price_to_earnings", "peRatio"):
            try:
                if hasattr(eq, "index") and key in eq.index:
                    row = eq.loc[key]
                    val = row.iloc[0] if hasattr(row, "iloc") else row
                    if pd.notna(val):
                        return float(val)
                elif hasattr(eq, "get") and eq.get(key) is not None:
                    return float(eq.get(key))
            except Exception:
                continue
        return None
