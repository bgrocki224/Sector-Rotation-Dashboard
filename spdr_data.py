"""
SPDR sector fundamentals -- free, issuer-published valuation data
==================================================================
State Street publishes a "Fund Characteristics" table on each Select Sector SPDR ETF's
product page, updated daily, including the fund's forward P/E ("Price/Earnings Ratio FY1":
price divided by forecast one-year EPS, harmonic-weighted across holdings). That is the
issuer's own number for the whole sector basket, so it needs no per-stock lookups and does
not touch Yahoo.

Like every snapshot source here it has no point-in-time history, so it is for LIVE scoring
only, never backtests (lookahead bias).

The scraper reads the page text, so it is deliberately defensive: it fails soft (returns an
error string per ticker rather than raising) and the caller falls back to Yahoo.
"""

from __future__ import annotations
import html as _html
import re
import urllib.request
from datetime import datetime
from typing import Optional

BASE = "https://www.ssga.com/us/en/intermediary/etfs/state-street-{slug}-select-sector-spdr-etf-{t}"
SLUGS = {
    "XLK": "technology", "XLF": "financial", "XLE": "energy", "XLV": "health-care",
    "XLY": "consumer-discretionary", "XLP": "consumer-staples", "XLI": "industrial",
    "XLB": "materials", "XLU": "utilities", "XLRE": "real-estate",
    "XLC": "communication-services",
}
LABEL = "Price/Earnings Ratio FY1"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36")


def _to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)   # drop scripts/styles
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)                       # drop tags
    return re.sub(r"\s+", " ", _html.unescape(raw))


def parse_forward_pe(page: str) -> tuple[Optional[float], Optional[str]]:
    """Returns (forward P/E, 'as of' date string) from a product page's text/HTML.
    Reads only the FUND characteristics block (not the Index block that follows)."""
    text = _to_text(page)
    m = re.search(r"Fund Characteristics\s+as of\s+([A-Z][a-z]{2}\s+\d{1,2}\s+\d{4})", text)
    as_of = m.group(1) if m else None
    start = m.end() if m else 0
    end = text.find("Index Characteristics", start)
    block = text[start:end if end != -1 else None]
    i = block.find(LABEL)
    if i == -1:
        return None, as_of
    # First standalone decimal after the label. Requiring a decimal point skips integers
    # that can appear in the row's definition text.
    num = re.search(r"(?<![\d.])(\d{1,3}\.\d{1,2})(?![\d.])", block[i + len(LABEL):])
    if not num:
        return None, as_of
    pe = float(num.group(1))
    return (pe if 0 < pe < 500 else None), as_of


def fetch_forward_pe(ticker: str, timeout: int = 20) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """(pe, as_of, error) for one sector ETF. Never raises."""
    slug = SLUGS.get(ticker)
    if not slug:
        return None, None, "no SPDR page known for this ticker"
    url = BASE.format(slug=slug, t=ticker.lower())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as e:  # network, HTTP error, block page...
        return None, None, f"{type(e).__name__}: {e}"
    pe, as_of = parse_forward_pe(page)
    if pe is None:
        return None, as_of, "page loaded but the 'Price/Earnings Ratio FY1' row wasn't found (layout change, or JS-rendered)"
    return pe, as_of, None


def days_old(as_of: Optional[str]) -> Optional[int]:
    if not as_of:
        return None
    try:
        return (datetime.now() - datetime.strptime(as_of, "%b %d %Y")).days
    except ValueError:
        return None
