"""
Market scanner — finds open Kalshi markets with potentially mispriced contracts.

Strategy (two-pass):
  Pass 1 — Near-term (primary): fetch markets closing within NEAR_DAYS directly
            via the /markets endpoint with max_ts filter. These are the liquid,
            actively-traded markets (economic indicators, sports, daily events).
  Pass 2 — Events (secondary): paginate events and fetch their markets to catch
            longer-horizon opportunities the first pass misses.

Mispricing signals:
  1. Spread arb   : yes_ask + no_ask < ARB_SIGNAL_THRESH (buy both sides, profit guaranteed)
  2. Price drift  : last traded price deviates from current mid (mean-reversion bet)
  3. Wide spread  : large bid-ask gap indicates thin book / stale quote
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any

import client as kalshi

logger = logging.getLogger("scanner")

# Cap concurrent Kalshi API calls — stays well under their rate limit
_SEMAPHORE = asyncio.Semaphore(8)

# How many days out to look in Pass 1 (near-term liquid markets)
NEAR_DAYS = 2
# Max markets to fetch in Pass 1 — caps the pagination before we time out
NEAR_MAX = 400
# Hard ceiling on close_time — never trade markets settling more than this far out.
# Applied to both scan passes so the events-based pass can't sneak in 2028+ markets.
MAX_HORIZON_SEC = 48 * 3600  # 48 hours
# Spread-arb signal threshold (scanner flags, autopilot decides whether to trade)
ARB_SIGNAL_THRESH = 98


def _to_cents(val: Any) -> float:
    try:
        return float(val) * 100
    except (TypeError, ValueError):
        return 0.0


def _parse_close_ts(close_time: object) -> float | None:
    """Return Unix timestamp from Kalshi's close_time (int seconds or ISO string)."""
    if close_time is None:
        return None
    if isinstance(close_time, (int, float)):
        return float(close_time)
    try:
        ct = datetime.fromisoformat(str(close_time).replace("Z", "+00:00"))
        return ct.timestamp()
    except Exception:
        return None


def _score_market(market: dict) -> dict | None:
    # ── Hard time ceiling ─────────────────────────────────────────────────────
    # Only trade markets closing within the next 48 hours (live / near-term).
    # This filters out 2028/2029/multi-year markets from both scan passes.
    close_ts = _parse_close_ts(market.get("close_time"))
    if close_ts is not None and close_ts > time.time() + MAX_HORIZON_SEC:
        return None

    yes_bid = _to_cents(market.get("yes_bid_dollars"))
    yes_ask = _to_cents(market.get("yes_ask_dollars"))
    no_bid  = _to_cents(market.get("no_bid_dollars"))
    no_ask  = _to_cents(market.get("no_ask_dollars"))
    last    = _to_cents(market.get("last_price_dollars"))

    # Need a real two-sided book
    if yes_bid <= 0 or yes_ask <= 0 or no_bid <= 0 or no_ask <= 0:
        return None

    # Skip markets where either side is near resolution — terrible risk/reward
    # At 70¢, max loss ≤ 2.3× max gain; above this the downside is disproportionate
    if yes_ask >= 70 or no_ask >= 70:
        return None
    if yes_ask <= 5 or no_ask <= 5:
        return None

    # Skip illiquid markets with huge bid-ask spreads (≥20¢) — entering is too costly
    if yes_ask - yes_bid >= 20:
        return None

    gap        = yes_ask + no_ask
    mid        = (yes_bid + yes_ask) / 2.0
    yes_spread = yes_ask - yes_bid
    signals: list[str] = []
    score   = 0.0

    # Signal 1: buying both sides is profitable at expiry
    if gap < ARB_SIGNAL_THRESH:
        signals.append(f"spread_arb gap={gap:.1f}¢ (yes@{yes_ask:.1f}+no@{no_ask:.1f})")
        score += (ARB_SIGNAL_THRESH - gap) * 3.0

    # Signal 2: last traded price has drifted from current mid
    if last > 0:
        drift = abs(last - mid)
        if drift > 5:
            signals.append(f"price_drift={drift:.1f}¢ (last={last:.1f}¢ mid={mid:.1f}¢)")
            score += drift * 1.5

    # Signal 3: wide bid-ask spread (thin book, potential inefficiency)
    if yes_spread > 15:
        signals.append(f"wide_spread={yes_spread:.1f}¢")
        score += yes_spread * 0.4

    if not signals:
        return None

    return {
        "ticker":        market.get("ticker"),
        "title":         market.get("title"),
        "event_ticker":  market.get("event_ticker"),
        "category":      market.get("category"),
        "yes_bid":       round(yes_bid, 1),
        "yes_ask":       round(yes_ask, 1),
        "no_bid":        round(no_bid, 1),
        "no_ask":        round(no_ask, 1),
        "last_price":    round(last, 1),
        "mid":           round(mid, 1),
        "spread_gap":    round(gap, 1),
        "score":         round(score, 2),
        "signals":       signals,
        "close_time":    market.get("close_time"),
        "yes_ask_size":  market.get("yes_ask_size_fp"),
        "yes_bid_size":  market.get("yes_bid_size_fp"),
    }


async def _fetch_page(max_ts: int | None = None, cursor: str | None = None) -> dict:
    async with _SEMAPHORE:
        return await kalshi.get_markets(
            status="open",
            limit=200,
            cursor=cursor,
            max_ts=max_ts,
        )


async def _fetch_markets_near_term() -> list[dict]:
    """Pass 1: markets closing within NEAR_DAYS, capped at NEAR_MAX to stay fast."""
    max_ts = int(time.time()) + NEAR_DAYS * 86400
    markets: list[dict] = []
    cursor: str | None = None

    while len(markets) < NEAR_MAX:
        data = await _fetch_page(max_ts=max_ts, cursor=cursor)
        page = data.get("markets", [])
        markets.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break

    markets = markets[:NEAR_MAX]
    logger.info(f"Pass 1 (≤{NEAR_DAYS}d): {len(markets)} markets fetched")
    return markets


async def _get_markets_for_event(event_ticker: str) -> list[dict]:
    """Fetch all open markets for one event, rate-limited."""
    async with _SEMAPHORE:
        try:
            data = await kalshi.get_markets(status="open", limit=200, event_ticker=event_ticker)
            return data.get("markets", [])
        except Exception as exc:
            logger.debug(f"Failed to fetch markets for {event_ticker}: {exc}")
            return []


async def _fetch_markets_via_events(max_events: int, category: str | None) -> list[dict]:
    """Pass 2: events-first scan for longer-horizon markets."""
    events: list[dict] = []
    cursor: str | None = None

    while len(events) < max_events:
        batch = min(200, max_events - len(events))
        async with _SEMAPHORE:
            data = await kalshi.get_events(limit=batch, cursor=cursor, category=category)
        page = data.get("events", [])
        events.extend(page)
        cursor = data.get("cursor")
        if not cursor or not page:
            break

    if not events:
        return []

    tasks = [_get_markets_for_event(e["event_ticker"]) for e in events]
    batches = await asyncio.gather(*tasks, return_exceptions=True)
    markets: list[dict] = []
    for b in batches:
        if isinstance(b, list):
            markets.extend(b)
    return markets


async def scan_markets(
    max_events: int = 100,
    min_score: float = 1.0,
    category: str | None = None,
    event_ticker: str | None = None,
) -> tuple[list[dict], int]:
    """
    Scan open Kalshi markets for mispricing signals.

    Returns (scored_results_sorted_by_score, total_markets_checked).
    """
    if event_ticker:
        markets = await _get_markets_for_event(event_ticker)
    else:
        # Run both passes concurrently
        near_task   = asyncio.create_task(_fetch_markets_near_term())
        events_task = asyncio.create_task(_fetch_markets_via_events(max_events, category))
        near_markets, event_markets = await asyncio.gather(near_task, events_task)

        # Deduplicate by ticker — near-term pass wins (more fresh data)
        seen: set[str] = set()
        markets = []
        for m in [*near_markets, *event_markets]:
            t = m.get("ticker")
            if t and t not in seen:
                seen.add(t)
                markets.append(m)

    results = []
    for m in markets:
        sig = _score_market(m)
        if sig and sig["score"] >= min_score:
            results.append(sig)

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info(
        f"Scan complete: {len(markets)} markets checked → "
        f"{len(results)} signals (min_score={min_score})"
    )
    return results, len(markets)


async def get_market_detail(ticker: str) -> dict:
    """Return full market info + orderbook snapshot."""
    market_data, orderbook_data = await asyncio.gather(
        kalshi.get_market(ticker),
        kalshi.get_orderbook(ticker, depth=10),
    )
    return {
        "market":    market_data.get("market", market_data),
        "orderbook": orderbook_data.get("orderbook", orderbook_data),
    }
