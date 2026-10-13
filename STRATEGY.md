# Sector Rotation Model — Strategy Documentation

## 1. Objective
Systematically rotate among U.S. sector ETFs on a monthly basis to:
- Overweight sectors with the strongest forward-looking relative outlook
- Reduce/avoid exposure to weakening sectors before major drawdowns
- Outperform a broad-market benchmark (SPY) on a risk-adjusted basis over a full cycle

This is a **relative-strength / relative-value rotation model**, not a market-timing model —
it decides *which* sectors to hold, and uses a market-level risk overlay to decide *how much*
total equity exposure to carry.

---

## 2. Investable Universe

Default universe = the 11 SPDR Select Sector ETFs (most liquid, purest sector-pure exposure,
long history, tight spreads) plus a benchmark and a defensive parking asset.

| Ticker | Sector                  |
|--------|--------------------------|
| XLK    | Technology               |
| XLF    | Financials                |
| XLE    | Energy                    |
| XLV    | Health Care               |
| XLI    | Industrials               |
| XLY    | Consumer Discretionary    |
| XLP    | Consumer Staples          |
| XLU    | Utilities                 |
| XLB    | Materials                 |
| XLRE   | Real Estate               |
| XLC    | Communication Services    |
| SPY    | Benchmark (not traded, used for relative-strength calc) |
| BIL/SHY| Defensive cash-parking sleeve when risk overlay is "risk-off" |

Optional extensions: equal-weight variants (RSP-style, e.g. RYT, RGI...) to reduce
mega-cap concentration distortion in cap-weighted sector ETFs (esp. XLK, XLC, XLY).

---

## 3. Factor Model

Each sector ETF gets a **composite score** each month, built from 6 factor families plus
one overlay. Every raw factor is cross-sectionally **z-scored** across the sectors each month
(a sector's score reflects how it ranks *relative to its peers*), blended with configurable
weights, and the blend is re-standardized so its scale doesn't depend on the weights.

Revised after the factor-research review (Doeswijk & van Vliet 2011; Conover, Jensen, Johnson &
Mercer 2008; Mill Street Research; S&P DJI; SSGA; Molchanov & Stangl):

| Factor family | What it captures | Backtestable? | Weight |
|---|---|---|---|
| **Momentum** | 12-1 / 6m / 3m return persistence plus acceleration | Yes | 0.25 |
| **Trend** | Price vs 50/200-day MA, MA slope, relative strength vs SPY, volume trend | Yes | 0.20 |
| **Macro regime** *(new)* | Sector tilt from yield-curve slope change, high-yield spread change, broad-dollar change and Fed stance (2yr yield change), mapped to sectors through `SECTOR_MACRO_LOADINGS` | Yes (real FRED history, 1-day lag) | 0.30 |
| **Earnings revisions** *(replaces analyst sentiment)* | Net analyst EPS-revision breadth across top holdings | No — live snapshot only, neutral in backtests | 0.15 |
| **Seasonality** | Sector's average return in the upcoming calendar month | Yes | 0.05 |
| **Fundamentals** | Average trailing P/E of top holdings (cheaper scores higher) | No — live snapshot only, neutral in backtests | 0.05 |

**Overbought overlay** *(replaces the standalone RSI "market sentiment" factor)*: after blending,
any sector with 14-day RSI above 75 has 0.5 standard deviations subtracted from its composite.
Standalone RSI has weak evidence; as a stretch filter on top of trend and momentum it is
defensible, and because it only needs prices it is fully backtestable.

**Why the weights moved:** the research found seasonality weakened after publication, static
valuation level did not predict sector returns (Doeswijk & van Vliet) and suffers long
drawdowns, and rating *levels* are largely priced in while estimate *revisions* carry the
signal. The business-cycle "clock" itself is weakly supported (Molchanov & Stangl), so macro is
implemented as continuous tilts, not four discrete phases.

**Important caveats:** `SECTOR_MACRO_LOADINGS` are directional starting points drawn from the
literature, not coefficients fitted to this model — validate them with the diagnostics tool.
Only 4 of the 6 weighted factors can be backtested, so roughly 20% of the live composite has
no historical validation.

*Short interest and a standalone calendar-effects factor were dropped earlier (no clean ETF-level
short-interest data; calendar flag had zero cross-sectional variance).*

### Composite score
```
score_i = standardize( Σ w_f × zscore_f(i) ) - overbought_penalty × [RSI_i > 75]
```
Weights live in `CONFIG["factor_weights"]`.
These are starting points — the framework is built so you can optimize/adjust weights via
walk-forward validation rather than a single in-sample fit (see §6).

---

## 4. Portfolio Construction

- **Rebalance frequency:** monthly (configurable to bi-weekly).
- **Selection:** hold the top N sectors by composite score (default N = 4–5 of 11).
- **Minimum diversification floor:** the portfolio never holds fewer than `min_sectors_held`
  (default 4) sectors at rebalance time, even if `top_n` is set lower or the hysteresis band
  would otherwise produce a smaller set — the constructor backfills with the next-best-ranked
  sectors to reach the floor. Note this floor applies at each scheduled rebalance; the
  trailing-stop rule below can still temporarily reduce the number of live positions between
  rebalances if a stop triggers intra-month (see the trade-off note there).
- **Weighting:** score-tilted — weight ∝ (score − min_score), floor at equal-weight if scores are close, capped at a max single-sector weight (default 30%).
- **Minimum holding / turnover control:** a sector already held is only sold if its rank falls
  below a lower threshold (e.g., drops out of top 6), not the instant it exits the top N. This
  hysteresis band reduces whipsaw/turnover.
- **Profit-taking / trailing stop:** if a held sector's price is > X% above its 50-day MA
  (overextension) or drawn down > Y% from its rolling high since entry, trim toward target
  weight or exit early rather than waiting for the next scheduled rebalance. *Trade-off:* this
  can temporarily drop the live sector count below `min_sectors_held` until the next rebalance
  rebuilds the portfolio — that's intentional (a stop exists to get you out of a name that's
  breaking down, not to be overridden by a diversification rule), but it means the 4-sector
  floor is a rebalance-time guarantee, not an every-single-day guarantee.
- **Cash/defensive sleeve:** if market-wide risk overlay (below) signals risk-off, shift
  a portion of the portfolio to BIL/SHY regardless of sector scores.

---

## 5. Risk Overlay (market-level, sits above sector selection)
A simple regime filter that scales *total* equity exposure (not sector choice):
- SPY price vs its 200-day MA (trend filter)
- Realized volatility regime (VIX level / percentile, or SPY realized vol)
- Breadth (% of the 11 sectors above their own 200-day MA)

Risk-off → cut gross equity exposure (e.g., 100% → 50–60%) and rotate the freed capital to
the defensive sleeve. This is the primary mechanism for "avoiding downside," since sector
rotation alone doesn't protect against broad market drawdowns (correlations go to 1).

---

## 6. Backtesting & Validation Notes
- Use **walk-forward** validation (train factor weights on a rolling window, test out-of-sample),
  not a single full-history optimization — full-history fits will overstate results.
- Include realistic **transaction costs and slippage** (ETF spreads are tight but not zero) and
  a monthly turnover cap.
- Benchmark against SPY buy-and-hold AND against an equal-weight-always-11-sectors baseline —
  the model needs to beat both, not just the market, to justify the added complexity/turnover.
- Watch for **data-availability survivorship**: XLRE and XLC only started in 2015/2018, so
  full-11-sector backtests only go back to late 2018 without adjustment.

## 7. Data Sourcing Notes (for factors without a free clean feed)
- **Price/volume/momentum/trend:** yfinance / any OHLCV provider — straightforward.
- **Macro regime:** free from FRED (T10Y2Y, BAMLH0A0HYM2, DTWEXBGS, DGS2) — no key needed.
- **Fundamentals & earnings revisions:** free yfinance proxies exist for live use only; real history needs a fundamentals/estimates API (e.g., Financial Modeling Prep,
  Refinitiv, Zacks, Bloomberg) aggregated up to sector-ETF level via holdings weights, since
  ETF-level "P/E" isn't a single ticker lookup.
- **Sentiment:** CBOE put/call ratio by sector where available, AAII/NAAIM survey for the
  market-wide regime component, ETF.com or provider fund-flow data for flow momentum.

The code ships with all of this behind a `DataProvider` interface with clean seams — price/
momentum/trend work out of the box with yfinance; the other factors have a documented plug-in
point and a neutral (zero-effect) fallback so the model still runs end-to-end without every
paid data feed wired up.
