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

Each sector ETF gets a **composite score** each month, built from 7 factor families.
Every raw factor is cross-sectionally **z-scored** across the 11 sectors each month
(so a sector's score reflects how it ranks *relative to its peers*, not in absolute terms),
then combined using configurable weights.

| Factor family        | What it captures                                   | Example signals |
|-----------------------|-----------------------------------------------------|------------------|
| **Momentum**          | Persistence of relative price trends                 | 1/3/6/12-month total return (12-1 skip-month convention), acceleration (3m vs 12m) |
| **Trend / price action** | Technical health, market activity                 | Price vs 50/200-day MA, MA slope, relative strength vs SPY, volume trend, ADX-style trend strength |
| **Seasonality**        | Historical calendar tendencies by sector             | Avg. sector return in this calendar month over trailing 10–15 years; "Sell in May" adjustment |
| **Calendar effects**   | Structural calendar patterns                         | Turn-of-month effect, pre/post FOMC drift, quad-witching week, presidential-cycle year |
| **Fundamentals**       | Valuation & earnings quality of the sector           | Forward P/E vs 5-yr average (relative), EPS growth estimates, margin trends (holdings-weighted) |
| **Analyst sentiment**  | Sell-side revisions momentum                          | 1m/3m EPS estimate revision breadth, upgrade/downgrade ratio, target-price revisions |
| **Market sentiment**   | Positioning & crowd behavior                          | Put/call ratio by sector, ETF fund-flow momentum, AAII/NAAIM overlay for market-wide risk regime |

*Short interest was considered but dropped: most sector ETFs don't have short interest data published in a clean, timely, comparable way (unlike single-name stocks), so the signal-to-noise on an ETF-level proxy wasn't worth the added complexity.*

### Composite score
```
score_i = Σ (w_f × zscore_f(i))   for f in factor families, i in sectors
```
Default starting weights (edit in `CONFIG["factor_weights"]`):
```
momentum:            0.25
trend:                0.15
seasonality:          0.10
calendar:             0.05
fundamentals:         0.15
analyst_sentiment:    0.15
market_sentiment:     0.15
```
These are starting points — the framework is built so you can optimize/adjust weights via
walk-forward validation rather than a single in-sample fit (see §6).

---

## 4. Portfolio Construction

- **Rebalance frequency:** monthly (configurable to bi-weekly).
- **Selection:** hold the top N sectors by composite score (default N = 4–5 of 11).
- **Weighting:** score-tilted — weight ∝ (score − min_score), floor at equal-weight if scores are close, capped at a max single-sector weight (default 30%).
- **Minimum holding / turnover control:** a sector already held is only sold if its rank falls
  below a lower threshold (e.g., drops out of top 6), not the instant it exits the top N. This
  hysteresis band reduces whipsaw/turnover.
- **Profit-taking / trailing stop:** if a held sector's price is > X% above its 50-day MA
  (overextension) or drawn down > Y% from its rolling high since entry, trim toward target
  weight or exit early rather than waiting for the next scheduled rebalance.
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
- **Fundamentals & analyst sentiment:** need a fundamentals API (e.g., Financial Modeling Prep,
  Refinitiv, Zacks, Bloomberg) aggregated up to sector-ETF level via holdings weights, since
  ETF-level "P/E" isn't a single ticker lookup.
- **Sentiment:** CBOE put/call ratio by sector where available, AAII/NAAIM survey for the
  market-wide regime component, ETF.com or provider fund-flow data for flow momentum.

The code ships with all of this behind a `DataProvider` interface with clean seams — price/
momentum/trend work out of the box with yfinance; the other factors have a documented plug-in
point and a neutral (zero-effect) fallback so the model still runs end-to-end without every
paid data feed wired up.
