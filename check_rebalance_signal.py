"""
Rebalance Signal Checker — for scheduled/cron execution
==========================================================
Lightweight alternative (or companion) to the Streamlit dashboard: run this on a schedule
(e.g. daily via cron) and it will notify you only when action is actually needed --
i.e. near/at a monthly rebalance date, or if a trailing-stop would trigger intraday logic
you want to know about sooner than the next scheduled rebalance.

Usage:
    python3 check_rebalance_signal.py                 # prints to stdout
    python3 check_rebalance_signal.py --slack-webhook <url>
    python3 check_rebalance_signal.py --email you@example.com --smtp-config smtp.json

Suggested cron entry (runs every weekday at 4:15pm ET, after market close):
    15 16 * * 1-5 cd /path/to/sector_rotation && python3 check_rebalance_signal.py --slack-webhook "$SLACK_WEBHOOK_URL"

State is persisted to `last_holdings.json` in the working directory so the script can tell
you what CHANGED since the last check, not just the current target weights.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from sector_rotation_model import (
    SECTOR_UNIVERSE, BENCHMARK, DEFENSIVE_ASSET, CONFIG,
    DataProvider, FactorEngine, CompositeScorer, PortfolioConstructor,
)
from extended_data_provider import YFinanceExtendedProvider

STATE_FILE = Path("last_holdings.json")
REBALANCE_WINDOW_DAYS = 2  # alert if within this many days of month-end (before or after)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(weights: dict, as_of: str):
    STATE_FILE.write_text(json.dumps({"as_of": as_of, "weights": weights}, indent=2))


def compute_target(start_date: str = "2015-01-01", use_extended: bool = True) -> tuple[pd.Timestamp, dict, pd.Series]:
    tickers = list(SECTOR_UNIVERSE.keys()) + [BENCHMARK, DEFENSIVE_ASSET]
    # Safe here: this script only ever asks for TODAY's signal, never a historical
    # backtest date, so the extended provider's snapshot-only limitation doesn't apply.
    provider_cls = YFinanceExtendedProvider if use_extended else DataProvider
    provider = provider_cls(tickers=tickers, start=start_date)
    prices = provider.load_prices()
    volume = provider.load_volume()
    sector_tickers = [t for t in SECTOR_UNIVERSE if t in prices.columns]

    engine = FactorEngine(prices, volume, provider)
    as_of = prices.index[-1]
    factors = engine.all_factors(as_of, sector_tickers)
    scorer = CompositeScorer(CONFIG["factor_weights"])
    scores = scorer.score(factors)

    constructor = PortfolioConstructor(CONFIG)
    prev_state = load_state()
    prev_holdings = set(prev_state.get("weights", {}).keys())
    target = constructor.target_weights(scores, prev_holdings)
    return as_of, target, scores


def is_near_rebalance(as_of: pd.Timestamp) -> bool:
    month_end = as_of + pd.offsets.MonthEnd(0)
    return abs((month_end - as_of).days) <= REBALANCE_WINDOW_DAYS


def format_message(as_of, target, scores, prev_weights) -> str:
    lines = [f"*Sector Rotation Update — {as_of.strftime('%Y-%m-%d')}*", ""]
    lines.append("Top ranked sectors:")
    for t, s in scores.sort_values(ascending=False).head(5).items():
        lines.append(f"  {t} ({SECTOR_UNIVERSE.get(t, t)}): score {s:+.2f}")
    lines.append("")
    lines.append("Target weights:")
    for t, w in sorted(target.items(), key=lambda x: -x[1]):
        prev_w = prev_weights.get(t, 0.0)
        delta = w - prev_w
        flag = " (NEW)" if prev_w == 0 else (f" ({delta:+.1%})" if abs(delta) > 0.01 else "")
        lines.append(f"  {t}: {w:.1%}{flag}")
    dropped = set(prev_weights) - set(target)
    if dropped:
        lines.append("")
        lines.append("Exited: " + ", ".join(sorted(dropped)))
    return "\n".join(lines)


def send_slack(webhook_url: str, message: str):
    import urllib.request
    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10)


def send_email(to_addr: str, smtp_config_path: str, subject: str, body: str):
    import smtplib
    from email.mime.text import MIMEText
    cfg = json.loads(Path(smtp_config_path).read_text())
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = cfg["from"]
    msg["To"] = to_addr
    with smtplib.SMTP(cfg["host"], cfg.get("port", 587)) as server:
        server.starttls()
        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["from"], [to_addr], msg.as_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slack-webhook", default=os.environ.get("SLACK_WEBHOOK_URL"))
    parser.add_argument("--email")
    parser.add_argument("--smtp-config", default="smtp.json")
    parser.add_argument("--force", action="store_true", help="Send update regardless of rebalance window")
    parser.add_argument("--no-extended", action="store_true",
                         help="Disable fundamentals/analyst/sentiment proxies (momentum/trend/seasonality/calendar only)")
    args = parser.parse_args()

    as_of, target, scores = compute_target(use_extended=not args.no_extended)
    prev_state = load_state()
    prev_weights = prev_state.get("weights", {})

    should_alert = args.force or is_near_rebalance(as_of) or set(target) != set(prev_weights)
    if not should_alert:
        print(f"[{as_of.date()}] No rebalance action needed. Holding current positions.")
        return

    message = format_message(as_of, target, scores, prev_weights)
    print(message)

    if args.slack_webhook:
        try:
            send_slack(args.slack_webhook, message)
            print("\n(sent to Slack)")
        except Exception as e:
            print(f"\n[WARN] Slack send failed: {e}", file=sys.stderr)

    if args.email:
        try:
            send_email(args.email, args.smtp_config, "Sector Rotation Update", message)
            print("(sent via email)")
        except Exception as e:
            print(f"[WARN] Email send failed: {e}", file=sys.stderr)

    save_state(target, str(as_of.date()))


if __name__ == "__main__":
    main()
