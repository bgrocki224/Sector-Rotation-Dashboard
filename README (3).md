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
factors out of the box. Fundamentals, analyst sentiment, and market sentiment run
in "neutral" mode (contribute 0 to the score) until you wire in a data feed — see below.

## Wiring in the remaining factors
As of the latest version, `extended_data_provider.py` (`YFinanceExtendedProvider`) wires
fundamentals, analyst sentiment, and market sentiment to **free yfinance-based proxies**
— no paid API key needed:
- **Fundamentals** — ETF-level trailing P/E, cross-sector relative valuation.
- **Analyst sentiment** — average analyst `recommendationMean` across each sector's top
  holdings.
- **Market sentiment** — 14-day RSI distance from neutral, as a positioning/stretch proxy.

These are proxies, not what a paid vendor (FMP, Zacks, Refinitiv) would give you — read
the docstrings in `extended_data_provider.py` for exactly what's approximated and its
limitations. Two things worth knowing:
- **Lookahead bias:** fundamentals/analyst data only exist as today's snapshot (no free
  historical time series), so they're only safe for **live/current** scoring, never for
  historical backtesting — using them across historical rebalance dates would silently
  feed today's data into decisions dated years ago. `streamlit_app.py` and
  `check_rebalance_signal.py` are already wired correctly: the backtest always uses the
  base `DataProvider` (momentum/trend/seasonality only); only the live "current
  scores" panel/signal uses the extended provider.
- **Rate limits:** the analyst-sentiment proxy fetches each sector's top ~8 holdings
  individually (~90-110 extra calls total). Holdings composition is cached for 7 days by
  default. If you hit Yahoo Finance rate limits (more likely on a shared host like
  Streamlit Community Cloud than locally), turn off the "Use live fundamentals/analyst/
  sentiment proxies" toggle in the dashboard sidebar, or pass `--no-extended` to
  `check_rebalance_signal.py`.

To swap in a real paid vendor instead, subclass `DataProvider` (or
`YFinanceExtendedProvider`) directly:
```python
class MyProvider(DataProvider):
    def get_analyst_sentiment(self, as_of):
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

