# Sector Rotation Model — Quickstart

## Files
- `STRATEGY.md` — methodology writeup (universe, factors, rotation rules, risk overlay)
- `sector_rotation_model.py` — the full implementation (data interface, factors, scoring,
  portfolio construction, risk overlay, backtester). Verified to run end-to-end (tested
  against synthetic price data in this environment, since this sandbox has no internet
  access to pull real market data).

## Run it with real data
```bash
pip install yfinance pandas numpy

python3 -c "
from sector_rotation_model import SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG, DataProvider, RotationBacktester

tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
provider = DataProvider(tickers=tickers, start='2015-01-01')
bt = RotationBacktester(provider, CONFIG)
result = bt.run()

print(result.stats)
result.equity_curve.to_csv('equity_curve.csv')
result.holdings_history.to_csv('holdings_history.csv')
"
```
This pulls live sector ETF prices via yfinance and backtests the momentum/trend/seasonality
factors, the macro regime factor (free FRED data, no key) and the RSI overbought overlay out of
the box. Fundamentals and earnings revisions only exist as live snapshots, so they are neutral
(contribute 0) in backtests — see below.

## Data sources for each factor
- **Momentum, trend, seasonality, RSI overbought overlay** — prices via yfinance.
- **Macro regime** — free FRED series fetched without an API key (`T10Y2Y`, `BAMLH0A0HYM2`,
  `DTWEXBGS`, `DGS2`), lagged one day. Real history, so it is **backtested**. If FRED can't be
  reached the factor silently goes neutral — the dashboard and alert script now print a visible
  warning when that happens, since macro is the highest-weighted factor.
- **Fundamentals and earnings revisions** — `extended_data_provider.py`
  (`YFinanceExtendedProvider`) uses free yfinance proxies for **live scoring only**:
  - *Fundamentals* — average trailing P/E of each sector's top holdings (5% weight).
  - *Earnings revisions* — net analyst EPS-revision breadth (up vs. down revisions in the last
    30 days) across top holdings, replacing the old static analyst-rating level (15% weight).

These two are proxies, not what a paid vendor (FMP, I/B/E/S, FactSet) would give you. Notes:
- **Lookahead bias:** both only exist as today's snapshot, so they're safe for live scoring
  but never for backtests — the backtest always uses the base `DataProvider`, where they are
  neutral. About 20% of the live composite therefore has no historical validation.
- **Schema risk:** yfinance's `eps_revisions` table has changed between versions; the code
  probes column names defensively and skips holdings it can't read, but verify it returns
  non-NaN values for your sectors on first run.
- **Rate limits:** the proxies fetch each sector's top ~8 holdings individually. Holdings are
  cached for 7 days. If you hit Yahoo limits (likelier on Streamlit Cloud), turn off the
  "Use live fundamentals/earnings-revision proxies" sidebar toggle, or pass `--no-extended` to
  `check_rebalance_signal.py`.

### Troubleshooting: fundamentals / earnings_revisions show 0 or blank
Neutral (0) is **expected** in three places: the backtest, the factor-attribution diagnostics, and
the live table when the sidebar proxy toggle is off. Neutral in the **live ranking table** with the
toggle on means the fetch failed. Both factors depend on the same first step — looking up each
sector ETF's top holdings via Yahoo (`funds_data.top_holdings`) — so one blocked or rate-limited
endpoint zeroes both at once. Streamlit Community Cloud shares IP addresses, which makes Yahoo
throttling likely.

What the code now does about it:
- Failures are **recorded, not swallowed**: the dashboard shows "fundamentals X/11, earnings
  revisions Y/11" with a warning, plus a "Live data health" panel listing the exact errors
  (`check_rebalance_signal.py` prints the same to stderr).
- If Yahoo's holdings lookup fails, a **static list of sector leaders** is used so the factors
  still compute; the health panel labels those sectors `static fallback`. It is approximate, not the
  ETF's true current weighting.
- Rate-limit errors (HTTP 429) are retried once; failed calls are no longer cached, and a live
  fetch with gaps is retried after 5 minutes instead of being kept for 4 hours.

If the health panel shows errors even after retries, the real fix is a different source: a paid
estimates/fundamentals API (e.g. Financial Modeling Prep, I/B/E/S, FactSet). Only a paid feed with
point-in-time history would also let these two factors be **backtested** — free yfinance cannot.

To swap in a real paid vendor instead, subclass `DataProvider` (or
`YFinanceExtendedProvider`) directly:
```python
class MyProvider(DataProvider):
    def get_earnings_revisions(self, as_of):
        # return a pandas Series indexed by ticker (e.g. XLK, XLF, ...)
        # e.g. net EPS revision breadth from your vendor, sector-weighted
        ...
        return my_series
```
Then pass `MyProvider(...)` into `RotationBacktester` instead of the base `DataProvider`
— a paid feed with real historical data is safe to use in the backtest too.

**On the `calendar` factor:** it was removed. It was deliberately applied uniformly across
all sectors (not sector-specific), so it had zero cross-sectional variance and zero actual
effect on the ranking once z-scored — a vestigial factor that overlapped conceptually with
`seasonality`, which already covers per-sector calendar tendencies properly. Its weight
was folded into `seasonality`.

**On the minimum-sector-count rule:** the portfolio now never holds fewer than
`CONFIG["min_sectors_held"]` (default 4) sectors at rebalance time — `PortfolioConstructor`
backfills with the next-best-ranked sectors if `top_n` or the hysteresis band would
otherwise produce a smaller set. This is a rebalance-time guarantee; the trailing-stop rule
can still temporarily drop the live count below 4 intra-month if a stop triggers (a stop
exists to protect against a name breaking down, and isn't overridden by the diversification
floor — see STRATEGY.md §4 for the full trade-off).

## Tuning
All knobs live in the `CONFIG` dict at the top of `sector_rotation_model.py`:
- `factor_weights` — relative importance of each factor family
- `top_n` / `hysteresis_rank` — how concentrated the portfolio is and how much whipsaw is tolerated
- `trailing_stop_pct` / `overextension_pct` — profit-taking / drawdown discipline
- `risk_off_equity_floor` — how defensive the market-level overlay gets in a downturn

## Validation before using real capital
1. Run walk-forward, not a single full-history fit (see STRATEGY.md §6).
2. Compare against SPY buy-and-hold AND an equal-weight-11-sectors baseline.
3. Add realistic transaction costs (already parameterized via `txn_cost_bps`) and check
   sensitivity to slippage assumptions.
4. Stress-test specific historical drawdowns (2018 Q4, 2020 COVID, 2022) individually —
   aggregate CAGR/Sharpe can hide a strategy that fails exactly when you need it not to.

This is a research and engineering framework, not a signal to trade on as-is, and nothing
here constitutes financial advice.

---

## Running Live: Dashboard & Alerts

Two complementary ways to "actively watch" the strategy, beyond one-off backtests:

### Option A — Interactive dashboard (`streamlit_app.py`)
A browser-based app: current sector rankings, target portfolio, factor breakdown table,
days-to-next-rebalance, and a live-updating backtest chart vs. SPY.

**Run it locally:**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`. Adjust factor weights and top-N live in the sidebar —
the dashboard recomputes rankings and the backtest on the fly.

**Host it so it's reachable from any browser (free tier available):**
1. Push this folder to a GitHub repo (needs `streamlit_app.py`, `sector_rotation_model.py`,
   `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at
   `streamlit_app.py`, deploy.
3. Data is cached for 4 hours (`CACHE_TTL_SECONDS` in the app) so it stays reasonably fresh
   without hammering the price feed on every page load — adjust that constant if you want
   fresher or less frequent pulls.

### Option B — Scheduled alerts (`check_rebalance_signal.py`)
For "notify me, don't make me check a dashboard": a script meant to run on a schedule
(cron, GitHub Actions, or any scheduler) that only speaks up when there's something to act
on — near a monthly rebalance date, or when target holdings actually change.

```bash
# one-off run, prints to terminal
python3 check_rebalance_signal.py

# with Slack alerts (create an Incoming Webhook in your Slack workspace)
python3 check_rebalance_signal.py --slack-webhook "https://hooks.slack.com/services/..."

# with email (needs an smtp.json with host/port/user/password/from)
python3 check_rebalance_signal.py --email you@example.com --smtp-config smtp.json
```

**Cron example** (weekdays, 4:15pm ET / after market close):
```
15 16 * * 1-5 cd /path/to/sector_rotation && python3 check_rebalance_signal.py --slack-webhook "$SLACK_WEBHOOK_URL"
```
It persists the last known target weights to `last_holdings.json` so each run can tell you
what *changed* (new entries, exits, weight shifts), not just repeat the full target every day.

### Which to use
- Want to eyeball rankings/backtest whenever you feel like it → **dashboard**.
- Want a push notification only when action is needed, and to leave it running unattended
  → **scheduled alert script**. The two aren't mutually exclusive — run both.

### What "live" does and doesn't mean here
This gives you a **signal generator you can check or be notified by**, not an auto-trading
system — no orders are placed anywhere. Actually rebalancing (buying/selling the ETFs) is a
manual step at your broker unless you separately wire this into a broker API (e.g. Alpaca,
Interactive Brokers) — which is a meaningful additional step involving real execution risk,
and worth building/testing carefully and separately from the signal logic itself.

---

## Diagnostics: Factor Attribution & Diversification Sensitivity

Two tools (`diagnostics.py`, used by both `streamlit_app.py`'s "🔬 Diagnostics" section and
the standalone `run_diagnostics.py`) that answer empirically, rather than by guessing, which
parts of the model are actually helping or hurting since a given backtest start date:

**Factor attribution:**
- *Leave-one-out* — runs the model with each factor's weight zeroed out in turn. If CAGR
  goes UP without a factor, it was a net drag; if CAGR goes DOWN without it, it was net
  additive.
- *Single-factor* — runs the model with ONLY one factor driving the ranking, to see its
  standalone performance.
- Important: `fundamentals` / `earnings_revisions` only have today's
  live snapshot (no free historical time series), so they're deliberately excluded from
  every backtest to avoid lookahead bias — meaning they're mathematically inert in this
  report by design. Expect their leave-one-out delta to be ~0 and their single-factor rows
  to be identical to each other (a tied ranking, not a real signal). That's confirmation
  the exclusion is working, not a bug.

**Diversification sensitivity:** a grid sweep over `top_n` (sectors held) x `max_weight`
(concentration cap), holding everything else fixed, so you can see empirically whether the
current diversification rules are costing performance in a monthly-rebalanced strategy —
useful context if you're reconsidering `CONFIG["max_weight"]` / `CONFIG["min_sectors_held"]`.

**Run it in the dashboard:** scroll to "🔬 Diagnostics" at the bottom of `streamlit_app.py`,
pick the tab, click the button. Both reuse the price data already loaded for the backtest
above, so — beyond that initial load — every variant is CPU-only, no extra network calls.

**Run it from the terminal:**
```bash
python3 run_diagnostics.py --start 2018-01-01
```
Prints a full report and saves `attribution_leave_one_out.csv`, `attribution_single_factor.csv`,
and `sensitivity_sweep.csv` for further analysis. Use `--top-n` / `--max-weight` to customize
the sweep grid, or `--skip-sweep` to run attribution only (faster).

---

## Diagnostics: Component Attribution & Half-Sample Stability

`momentum` and `trend` are each an **average of sub-signals** (momentum: 12-1m, 6m, 3m return and
acceleration; trend: price vs 50d/200d MA, MA slope, relative strength, volume trend). Two things
worth knowing about the default behaviour:

- The sub-signals are averaged **raw**, in different units, so the one with the widest numeric range
  (12-month return, or the volume-surge ratio in trend) dominates its factor.
- Momentum's acceleration term is `r3 - r6`, so the 6-month return **cancels out algebraically**:
  momentum = (12-1m return + 2 x 3-month return) / 4.

New tools to test whether that matters:
- `CONFIG["standardize_components"]` (default `False`, which reproduces earlier results exactly).
  When `True`, each sub-signal is z-scored across sectors before averaging. Sidebar checkbox in the
  dashboard; `--standardize` in `run_diagnostics.py`.
- `CONFIG["momentum_components"]` / `CONFIG["trend_components"]` choose which sub-signals are used.
- **Component Attribution** tab / `attribution_components.csv`: "without X" and "only X" runs for
  every sub-signal.
- **`delta_h1` / `delta_h2`** columns on the leave-one-out and component tables: the CAGR effect in
  the first vs second half of the sample. Opposite signs mean the effect is probably noise.

Everything here is in-sample. A factor that looks like a drag over one period can be the one that
saves you in the next; prefer changes that are consistent across both halves.

