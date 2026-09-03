"""
Sector Rotation Model
======================
A monthly, multi-factor sector rotation framework for the 11 SPDR Select Sector ETFs.

Pipeline:
    Prices/Fundamentals/Sentiment data
        -> FactorEngine (raw factor calcs)
        -> CompositeScorer (cross-sectional z-score + weighted blend)
        -> PortfolioConstructor (top-N selection, score-tilted weights, hysteresis)
        -> RiskOverlay (market-wide regime filter scales total equity exposure)
        -> Backtester (walk the calendar month by month, apply costs, track performance)

Requires: pandas, numpy. Optional: yfinance (for the built-in live price loader).

See STRATEGY.md for the methodology writeup. See CONFIG below for all tunable parameters.

NOTE: This is a research/engineering framework, not investment advice. Validate thoroughly
(out-of-sample / walk-forward) before using with real capital.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# 1. UNIVERSE & CONFIG
# ---------------------------------------------------------------------------

SECTOR_UNIVERSE = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
BENCHMARK = "SPY"
DEFENSIVE_ASSET = "BIL"  # 1-3mo T-bill ETF used as the risk-off parking sleeve

CONFIG = {
    "rebalance_freq": "ME",         # 'ME' month-end, could support 'W' etc. (pandas >= 2.2 offset alias)
    "top_n": 4,                     # sectors held at full conviction
    "min_sectors_held": 4,          # hard floor: never hold fewer than this many sectors at once
    "hysteresis_rank": 6,           # a held sector isn't sold until it falls below this rank
    "max_weight": 0.35,             # cap on any single sector's portfolio weight
    "min_weight": 0.05,             # floor once a sector is included, else drop it
    "trailing_stop_pct": 0.12,      # exit early if -12% from the post-entry high
    "overextension_pct": 0.20,      # trim if price is 20%+ above 50d MA (mean-reversion risk)
    "txn_cost_bps": 5,              # round-trip cost assumption per rebalance leg, in bps
    "risk_overlay_enabled": True,
    "risk_off_equity_floor": 0.50,  # min total equity exposure retained even in risk-off
    "factor_weights": {
        "momentum": 0.25,
        "trend": 0.15,
        "seasonality": 0.15,
        "fundamentals": 0.15,
        "analyst_sentiment": 0.15,
        "market_sentiment": 0.15,
    },
}


# ---------------------------------------------------------------------------
# 2. DATA PROVIDER INTERFACE
# ---------------------------------------------------------------------------

class DataProvider:
    """
    Central data access point. Price data has a working default loader (yfinance).
    Fundamentals / analyst sentiment / market sentiment are pluggable: wire in
    your vendor of choice by overriding the relevant `get_*` method. Each has a neutral
    fallback (returns None / empty) so the pipeline still runs without every feed present
    -- that factor family will simply contribute zero information (z-score of 0) until wired.
    """

    def __init__(self, tickers: list[str], start: str, end: Optional[str] = None):
        self.tickers = tickers
        self.start = start
        self.end = end
        self._prices: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None

    # ---- Prices (working default via yfinance) ----
    def load_prices(self) -> pd.DataFrame:
        """Returns adjusted close prices, columns=tickers, index=dates."""
        if self._prices is not None:
            return self._prices
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError(
                "yfinance not installed. `pip install yfinance`, or override "
                "DataProvider.load_prices() with your own price feed."
            ) from e

        raw = yf.download(
            self.tickers, start=self.start, end=self.end,
            auto_adjust=True, progress=False, group_by="ticker",
        )
        # yfinance returns a MultiIndex when multiple tickers are requested
        if isinstance(raw.columns, pd.MultiIndex):
            close = pd.DataFrame({t: raw[t]["Close"] for t in self.tickers if t in raw.columns.levels[0]})
            vol = pd.DataFrame({t: raw[t]["Volume"] for t in self.tickers if t in raw.columns.levels[0]})
        else:
            close = raw[["Close"]].rename(columns={"Close": self.tickers[0]})
            vol = raw[["Volume"]].rename(columns={"Volume": self.tickers[0]})
        self._prices = close.dropna(how="all")
        self._volume = vol.reindex(self._prices.index)
        return self._prices

    def load_volume(self) -> pd.DataFrame:
        if self._volume is None:
            self.load_prices()
        return self._volume

    # ---- Pluggable factor feeds (override these with a real vendor) ----
    def get_fundamentals(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Should return a Series indexed by ticker: e.g. forward P/E relative to its
        own 5yr average (lower/cheaper = higher score should be applied downstream),
        plus EPS growth estimate. Return None if unavailable (-> neutral factor)."""
        return None

    def get_analyst_sentiment(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Series indexed by ticker: e.g. net EPS estimate revision breadth (upgrades
        minus downgrades / total estimates) over the trailing 1-3 months."""
        return None

    def get_market_sentiment(self, as_of: pd.Timestamp) -> Optional[pd.Series]:
        """Series indexed by ticker: e.g. inverse put/call ratio z-score, or ETF flow
        momentum (net creations - redemptions, normalized by AUM)."""
        return None


# ---------------------------------------------------------------------------
# 3. FACTOR ENGINE
# ---------------------------------------------------------------------------

class FactorEngine:
    """Computes raw factor values for every sector as of a given rebalance date,
    using only data available up to that date (no lookahead)."""

    def __init__(self, prices: pd.DataFrame, volume: pd.DataFrame, provider: DataProvider):
        self.prices = prices
        self.volume = volume
        self.provider = provider

    def _ret(self, px: pd.Series, days: int) -> float:
        if len(px) <= days:
            return np.nan
        return px.iloc[-1] / px.iloc[-1 - days] - 1

    def momentum(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.Series:
        """12-1 month momentum (skip most recent month to avoid short-term reversal),
        blended with 3m and 6m momentum, plus a simple acceleration term."""
        out = {}
        hist = self.prices.loc[:as_of]
        for t in tickers:
            px = hist[t].dropna()
            if len(px) < 260:
                out[t] = np.nan
                continue
            r12_1 = px.iloc[-22] / px.iloc[-252] - 1   # 12m return, skipping last month
            r6 = self._ret(px, 126)
            r3 = self._ret(px, 63)
            accel = r3 - r6  # positive = momentum accelerating recently
            out[t] = np.nanmean([r12_1, r6, r3, accel])
        return pd.Series(out, name="momentum")

    def trend(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.Series:
        """Price vs 50/200d MA, MA slope, relative strength vs benchmark, volume trend."""
        out = {}
        hist_px = self.prices.loc[:as_of]
        hist_vol = self.volume.loc[:as_of] if self.volume is not None else None
        bench = hist_px[BENCHMARK] if BENCHMARK in hist_px.columns else None
        for t in tickers:
            px = hist_px[t].dropna()
            if len(px) < 200:
                out[t] = np.nan
                continue
            ma50 = px.rolling(50).mean().iloc[-1]
            ma200 = px.rolling(200).mean().iloc[-1]
            ma200_prev = px.rolling(200).mean().iloc[-21]
            px_vs_ma50 = px.iloc[-1] / ma50 - 1
            px_vs_ma200 = px.iloc[-1] / ma200 - 1
            ma_slope = ma200 / ma200_prev - 1
            rel_strength = np.nan
            if bench is not None and len(bench.dropna()) >= 126:
                rel_strength = (px.iloc[-1] / px.iloc[-126]) / (bench.iloc[-1] / bench.iloc[-126]) - 1
            vol_trend = np.nan
            if hist_vol is not None and t in hist_vol.columns:
                v = hist_vol[t].dropna()
                if len(v) >= 60:
                    vol_trend = v.iloc[-20:].mean() / v.iloc[-60:-20].mean() - 1
            out[t] = np.nanmean([px_vs_ma50, px_vs_ma200, ma_slope, rel_strength, vol_trend])
        return pd.Series(out, name="trend")

    def seasonality(self, as_of: pd.Timestamp, tickers: list[str], lookback_years: int = 15) -> pd.Series:
        """Average historical return in the *upcoming* calendar month, over trailing N years."""
        out = {}
        target_month = (as_of.month % 12) + 1  # the month we're rotating INTO
        hist_px = self.prices.loc[:as_of]
        monthly = hist_px.resample("ME").last()
        monthly_ret = monthly.pct_change()
        cutoff = as_of - pd.DateOffset(years=lookback_years)
        window = monthly_ret.loc[cutoff:as_of]
        for t in tickers:
            if t not in window.columns:
                out[t] = np.nan
                continue
            same_month = window[t][window.index.month == target_month]
            out[t] = same_month.mean() if len(same_month) >= 3 else np.nan
        return pd.Series(out, name="seasonality")

    def fundamentals(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.Series:
        s = self.provider.get_fundamentals(as_of)
        if s is None:
            return pd.Series({t: 0.0 for t in tickers}, name="fundamentals")
        return s.reindex(tickers).rename("fundamentals")

    def analyst_sentiment(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.Series:
        s = self.provider.get_analyst_sentiment(as_of)
        if s is None:
            return pd.Series({t: 0.0 for t in tickers}, name="analyst_sentiment")
        return s.reindex(tickers).rename("analyst_sentiment")

    def market_sentiment(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.Series:
        s = self.provider.get_market_sentiment(as_of)
        if s is None:
            return pd.Series({t: 0.0 for t in tickers}, name="market_sentiment")
        return s.reindex(tickers).rename("market_sentiment")

    def all_factors(self, as_of: pd.Timestamp, tickers: list[str]) -> pd.DataFrame:
        fns = {
            "momentum": self.momentum,
            "trend": self.trend,
            "seasonality": self.seasonality,
            "fundamentals": self.fundamentals,
            "analyst_sentiment": self.analyst_sentiment,
            "market_sentiment": self.market_sentiment,
        }
        cols = {name: fn(as_of, tickers) for name, fn in fns.items()}
        return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# 4. COMPOSITE SCORER
# ---------------------------------------------------------------------------

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() < 2 or s.std(skipna=True) == 0 or np.isnan(s.std(skipna=True)):
        return pd.Series(0.0, index=s.index)
    z = (s - s.mean(skipna=True)) / s.std(skipna=True)
    return z.fillna(0.0)


class CompositeScorer:
    def __init__(self, weights: dict[str, float]):
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}  # normalize

    def score(self, factor_df: pd.DataFrame) -> pd.Series:
        z = factor_df.apply(zscore, axis=0)
        composite = sum(z[f] * w for f, w in self.weights.items() if f in z.columns)
        return composite.rename("score").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 5. PORTFOLIO CONSTRUCTOR
# ---------------------------------------------------------------------------

def bounded_normalize(raw: pd.Series, lower: float, upper: float, tol: float = 1e-9,
                       max_iter: int = 200) -> pd.Series:
    """Allocates weights summing to 1.0 subject to lower <= w_i <= upper for every i,
    staying as proportional to `raw` as the bounds allow (a box-constrained projection
    onto the simplex).

    A naive clip-then-renormalize (clip each value to [lower, upper], then divide by
    the new sum) does NOT reliably satisfy the bounds: pushing several small values up
    to `lower` simultaneously can push the total above 1.0, and renormalizing back down
    drags every value -- including the ones just raised to the floor -- back below it.

    This uses iterative clip-and-redistribute: clip to the bounds, measure how far the
    total is from 1.0, and spread that difference proportionally across only the
    values NOT currently pinned at a bound, then repeat. Each pass makes a small
    correction and re-clips, so it converges to a feasible fixed point rather than
    over-correcting in one shot (which was the bug in an earlier, one-shot version of
    this function).

    If the bounds are infeasible for the given count (e.g. lower * n > 1, too many
    names for the floor to be satisfiable at once, or upper * n < 1), falls back to
    equal weight.
    """
    idx = list(raw.index)
    n = len(idx)
    if n == 0:
        return pd.Series(dtype=float)
    if n * lower > 1 + tol or n * upper < 1 - tol:
        return pd.Series(1.0 / n, index=idx)

    total_raw = raw.sum()
    w = (raw / total_raw).astype(float) if total_raw > 0 else pd.Series(1.0 / n, index=idx)

    for _ in range(max_iter):
        w = w.clip(lower=lower, upper=upper)
        diff = w.sum() - 1.0
        if abs(diff) < tol:
            break
        at_lower = w <= lower + tol
        at_upper = w >= upper - tol
        free_mask = ~(at_lower | at_upper)
        if not free_mask.any():
            # everyone is pinned at a bound on this pass -- rescale and let the next
            # iteration's clip re-evaluate who's actually free (rescaling can un-pin
            # a value that only looked pinned due to the current overall scale).
            w = w / w.sum()
            continue
        free_sum = w[free_mask].sum()
        if free_sum > 0:
            w[free_mask] = w[free_mask] - diff * (w[free_mask] / free_sum)
        else:
            w[free_mask] = w[free_mask] - diff / free_mask.sum()

    return w


class PortfolioConstructor:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def target_weights(self, scores: pd.Series, current_holdings: set[str]) -> dict[str, float]:
        scores = scores.dropna()  # defensive: a NaN composite score shouldn't rank/weight
        ranked = scores.sort_values(ascending=False)
        ranks = {t: i + 1 for i, t in enumerate(ranked.index)}
        top_n = self.cfg["top_n"]
        hysteresis = self.cfg["hysteresis_rank"]
        min_sectors = self.cfg.get("min_sectors_held", top_n)

        keep = set()
        for t in ranked.index[:top_n]:
            keep.add(t)
        # hysteresis: retain existing holdings that haven't fallen too far in rank
        for t in current_holdings:
            if t in ranks and ranks[t] <= hysteresis:
                keep.add(t)

        # hard floor: never hold fewer than min_sectors, even if top_n/hysteresis
        # would otherwise produce a smaller set -- backfill with the next-best-ranked
        # sectors not already included.
        if len(keep) < min_sectors:
            for t in ranked.index:
                if len(keep) >= min_sectors:
                    break
                keep.add(t)

        if not keep:
            return {}

        sub = ranked.loc[list(keep)]
        shifted = sub - sub.min() + 1e-6  # ensure positive weights, scaled by relative score
        raw_w = shifted / shifted.sum()

        # Enforce BOTH bounds via bounded water-filling: every included sector gets
        # at least min_weight and at most max_weight, staying proportional to the
        # score-tilt as much as the bounds allow. (A naive clip-then-renormalize
        # doesn't reliably hold when multiple sectors hit the floor simultaneously --
        # see bounded_normalize's docstring.)
        final = bounded_normalize(raw_w, self.cfg["min_weight"], self.cfg["max_weight"])
        return final.to_dict()


# ---------------------------------------------------------------------------
# 6. RISK OVERLAY (market-wide exposure scaling)
# ---------------------------------------------------------------------------

class RiskOverlay:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def equity_exposure(self, as_of: pd.Timestamp, prices: pd.DataFrame, sector_tickers: list[str]) -> float:
        if not self.cfg["risk_overlay_enabled"]:
            return 1.0
        hist = prices.loc[:as_of]
        if BENCHMARK not in hist.columns or len(hist[BENCHMARK].dropna()) < 200:
            return 1.0
        spy = hist[BENCHMARK].dropna()
        trend_ok = spy.iloc[-1] > spy.rolling(200).mean().iloc[-1]

        breadth = np.nan
        avail = [t for t in sector_tickers if t in hist.columns]
        if avail:
            above_ma = []
            for t in avail:
                px = hist[t].dropna()
                if len(px) >= 200:
                    above_ma.append(px.iloc[-1] > px.rolling(200).mean().iloc[-1])
            if above_ma:
                breadth = np.mean(above_ma)

        floor = self.cfg["risk_off_equity_floor"]
        if trend_ok and (np.isnan(breadth) or breadth >= 0.4):
            return 1.0
        elif trend_ok:
            return max(floor, 0.75)
        else:
            return floor


# ---------------------------------------------------------------------------
# 7. BACKTESTER
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    holdings_history: pd.DataFrame
    turnover: pd.Series
    stats: dict = field(default_factory=dict)


class RotationBacktester:
    def __init__(self, provider: DataProvider, cfg: dict = CONFIG):
        self.provider = provider
        self.cfg = cfg
        self.scorer = CompositeScorer(cfg["factor_weights"])
        self.constructor = PortfolioConstructor(cfg)
        self.risk_overlay = RiskOverlay(cfg)

    @staticmethod
    def _eligible_tickers(prices: pd.DataFrame, tickers: list[str], as_of: pd.Timestamp,
                           min_history_days: int = 260) -> list[str]:
        """Excludes tickers that don't yet have enough trading history as of this date
        -- e.g. XLRE (inception Oct 2015) and XLC (inception Jun 2018) have no price
        data at all for a backtest starting Jan 2015. Without this filter, a young
        ETF's missing data produces NaN factor scores that z-score to a neutral 0,
        which can look artificially attractive next to genuinely weak established
        sectors and get "selected" before the ETF existed -- and once a NaN price
        enters the daily-return calculation, it multiplies through the equity curve
        forever after, which is what was producing NaN CAGR."""
        eligible = []
        hist = prices.loc[:as_of]
        for t in tickers:
            if t not in hist.columns:
                continue
            px = hist[t].dropna()
            if len(px) >= min_history_days and as_of in hist.index and pd.notna(hist.loc[as_of, t]):
                eligible.append(t)
        return eligible

    def run(self) -> BacktestResult:
        prices = self.provider.load_prices()
        volume = self.provider.load_volume()
        sector_tickers = [t for t in SECTOR_UNIVERSE if t in prices.columns]
        engine = FactorEngine(prices, volume, self.provider)

        rebal_dates = prices.resample(self.cfg["rebalance_freq"]).last().index
        rebal_dates = rebal_dates[rebal_dates >= prices.index[0] + pd.Timedelta(days=380)]  # warmup for 200d/12m factors

        equity = 1.0
        equity_curve = {}
        holdings_hist = []
        turnover_hist = {}
        current_weights: dict[str, float] = {}
        entry_high: dict[str, float] = {}

        daily_index = prices.index
        for i, rdate in enumerate(rebal_dates):
            if rdate not in daily_index:
                nearest = daily_index[daily_index <= rdate]
                if len(nearest) == 0:
                    continue
                rdate = nearest[-1]

            eligible = self._eligible_tickers(prices, sector_tickers, rdate)
            factors = engine.all_factors(rdate, eligible)
            scores = self.scorer.score(factors)
            new_weights = self.constructor.target_weights(scores, set(current_weights.keys()))

            equity_exp = self.risk_overlay.equity_exposure(rdate, prices, sector_tickers)
            scaled_weights = {t: w * equity_exp for t, w in new_weights.items()}

            all_tickers = set(current_weights) | set(scaled_weights)
            turnover = sum(abs(scaled_weights.get(t, 0) - current_weights.get(t, 0)) for t in all_tickers)
            cost = turnover * (self.cfg["txn_cost_bps"] / 10000)
            turnover_hist[rdate] = turnover

            current_weights = scaled_weights
            entry_high = {t: prices.loc[rdate, t] for t in current_weights}
            holdings_hist.append({"date": rdate, **current_weights, "equity_exposure": equity_exp})
            equity *= (1 - cost)
            equity_curve[rdate] = equity

            next_rdate = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else daily_index[-1]
            period_days = daily_index[(daily_index > rdate) & (daily_index <= next_rdate)]
            prev_px = prices.loc[rdate]
            for d in period_days:
                px = prices.loc[d]
                # profit-taking / trailing-stop check (simplified: applied end-of-day)
                for t in list(current_weights.keys()):
                    if t in entry_high:
                        entry_high[t] = max(entry_high[t], px[t])
                        dd = px[t] / entry_high[t] - 1
                        if dd <= -self.cfg["trailing_stop_pct"]:
                            current_weights[t] = 0.0  # exit; freed weight sits in cash till next rebal
                day_ret = 0.0
                for t, w in current_weights.items():
                    if w <= 0 or t not in px.index:
                        continue
                    if pd.isna(px[t]) or pd.isna(prev_px[t]):
                        continue  # skip a missing data point rather than poisoning the whole equity curve with NaN
                    day_ret += w * (px[t] / prev_px[t] - 1)
                equity *= (1 + day_ret)
                equity_curve[d] = equity
                prev_px = px

        ec = pd.Series(equity_curve).sort_index()
        hh = pd.DataFrame(holdings_hist).set_index("date") if holdings_hist else pd.DataFrame()
        to = pd.Series(turnover_hist).sort_index()

        stats = self._compute_stats(ec, prices)
        return BacktestResult(equity_curve=ec, holdings_history=hh, turnover=to, stats=stats)

    @staticmethod
    def _compute_stats(ec: pd.Series, prices: pd.DataFrame) -> dict:
        if len(ec) < 2:
            return {}
        rets = ec.pct_change().dropna()
        years = (ec.index[-1] - ec.index[0]).days / 365.25
        cagr = ec.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
        vol = rets.std() * np.sqrt(252)
        sharpe = (rets.mean() * 252) / vol if vol > 0 else np.nan
        roll_max = ec.cummax()
        dd = (ec / roll_max - 1)
        max_dd = dd.min()

        bench_stats = {}
        if BENCHMARK in prices.columns:
            bench = prices[BENCHMARK].reindex(ec.index).dropna()
            bench = bench / bench.iloc[0]
            b_rets = bench.pct_change().dropna()
            b_years = (bench.index[-1] - bench.index[0]).days / 365.25
            bench_stats = {
                "benchmark_cagr": bench.iloc[-1] ** (1 / b_years) - 1 if b_years > 0 else np.nan,
                "benchmark_max_dd": (bench / bench.cummax() - 1).min(),
                "benchmark_vol": b_rets.std() * np.sqrt(252),
            }

        return {
            "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": max_dd,
            **bench_stats,
        }


# ---------------------------------------------------------------------------
# 8. EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    provider = DataProvider(tickers=tickers, start="2015-01-01")
    bt = RotationBacktester(provider, CONFIG)
    result = bt.run()

    print("=== Performance Summary ===")
    for k, v in result.stats.items():
        print(f"{k:>18}: {v:.2%}" if isinstance(v, float) else f"{k:>18}: {v}")

    print("\n=== Last Rebalance Holdings ===")
    if not result.holdings_history.empty:
        print(result.holdings_history.iloc[-1].dropna())
