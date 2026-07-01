"""
Thin async Kalshi REST client (API v2).

Set KALSHI_ENV=demo to hit the demo environment instead of production.
"""
import os
import httpx
from auth import build_auth_headers

_ENV = os.getenv("KALSHI_ENV", "prod").lower()

if _ENV == "demo":
    BASE = "https://demo-api.kalshi.co/trade-api/v2"
    BASE_PATH = "/trade-api/v2"
else:
    BASE = "https://api.elections.kalshi.com/trade-api/v2"
    BASE_PATH = "/trade-api/v2"


import asyncio
import logging

_log = logging.getLogger("kalshi.client")
_HTTP = httpx.AsyncClient(timeout=15)   # shared client — no per-request TLS handshake


async def _request_with_retry(method: str, url: str, headers: dict, **kwargs) -> httpx.Response:
    """Retry up to 4 times on 429; log full body on other 4xx/5xx errors."""
    delay = 1.0
    for attempt in range(5):
        r = await _HTTP.request(method, url, headers=headers, **kwargs)
        if r.status_code == 429:
            _log.warning(f"429 rate-limit — waiting {delay:.1f}s (attempt {attempt+1})")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 16)
            continue
        if not r.is_success:
            _log.error(
                f"HTTP {r.status_code} {r.reason_phrase} on {method} {url} "
                f"— body: {r.text[:400]}"
            )
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


async def _get(path: str, params: dict | None = None) -> dict:
    headers = build_auth_headers("GET", BASE_PATH + path)
    r = await _request_with_retry("GET", BASE + path, headers=headers, params=params)
    return r.json()


async def _post(path: str, body: dict) -> dict:
    headers = build_auth_headers("POST", BASE_PATH + path)
    r = await _request_with_retry("POST", BASE + path, headers=headers, json=body)
    return r.json()


async def _delete(path: str) -> dict:
    headers = build_auth_headers("DELETE", BASE_PATH + path)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.delete(BASE + path, headers=headers)
        r.raise_for_status()
        return r.json()


# ── Events ───────────────────────────────────────────────────────────────────

async def get_events(
    limit: int = 200,
    cursor: str | None = None,
    category: str | None = None,
    status: str = "open",
) -> dict:
    params: dict = {"limit": limit, "status": status}
    if cursor:
        params["cursor"] = cursor
    if category:
        params["category"] = category
    return await _get("/events", params=params)


# ── Markets ──────────────────────────────────────────────────────────────────

async def get_markets(
    status: str = "open",
    limit: int = 200,
    cursor: str | None = None,
    event_ticker: str | None = None,
    series_ticker: str | None = None,
    max_ts: int | None = None,   # unix seconds — only markets closing before this time
    min_ts: int | None = None,   # unix seconds — only markets closing after this time
) -> dict:
    params: dict = {"status": status, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    if event_ticker:
        params["event_ticker"] = event_ticker
    if series_ticker:
        params["series_ticker"] = series_ticker
    if max_ts is not None:
        params["max_ts"] = max_ts
    if min_ts is not None:
        params["min_ts"] = min_ts
    return await _get("/markets", params=params)


async def get_market(ticker: str) -> dict:
    return await _get(f"/markets/{ticker}")


async def get_orderbook(ticker: str, depth: int = 5) -> dict:
    return await _get(f"/markets/{ticker}/orderbook", params={"depth": depth})


# ── Orders (V2 single-book API) ────────────────────────────────────────────────
# New V2 endpoint: POST /portfolio/events/orders
# Side is "bid" (buy YES) or "ask" (sell YES / buy NO).
# Price is YES-denominated in dollars ("0.2300"), NOT cents.
# Required: time_in_force, self_trade_prevention_type.

def _to_v2_side_and_price(
    side: str,       # "yes" or "no"
    action: str,     # "buy" or "sell"
    yes_price: int | None,  # cents; for NO buy this is no_ask cents, NOT a YES price
    order_type: str,
) -> tuple[str, str, str]:
    """Return (book_side, price_str, time_in_force) for the V2 API."""
    if order_type == "market":
        # Market order → IOC with an aggressive price to guarantee fill
        if action == "buy" and side == "yes":
            return "bid", "0.9900", "immediate_or_cancel"
        if action == "buy" and side == "no":
            return "ask", "0.0100", "immediate_or_cancel"
        if action == "sell" and side == "yes":
            return "ask", "0.0100", "immediate_or_cancel"
        # sell no = buy yes
        return "bid", "0.9900", "immediate_or_cancel"

    # Limit order: translate cents price to dollar string
    p = yes_price or 50
    if action == "buy":
        if side == "yes":
            # bid at YES price
            return "bid", f"{p / 100:.4f}", "good_till_canceled"
        else:
            # ask: sell YES at (100 - no_ask) = implied YES price
            yes_equiv = 100 - p
            return "ask", f"{yes_equiv / 100:.4f}", "good_till_canceled"
    else:  # sell
        if side == "yes":
            return "ask", f"{p / 100:.4f}", "good_till_canceled"
        else:
            # sell no = buy yes; bid at (100 - no_price)
            yes_equiv = 100 - p
            return "bid", f"{yes_equiv / 100:.4f}", "good_till_canceled"


async def place_order(
    ticker: str,
    side: str,        # "yes" or "no"
    action: str,      # "buy" or "sell"
    order_type: str,  # "limit" or "market"
    count: int,
    yes_price: int | None = None,   # cents (1–99); for NO side this is the NO price
    client_order_id: str | None = None,
) -> dict:
    book_side, price_str, tif = _to_v2_side_and_price(side, action, yes_price, order_type)
    body: dict = {
        "ticker": ticker,
        "side": book_side,
        "count": f"{count}.00",
        "price": price_str,
        "time_in_force": tif,
        "self_trade_prevention_type": "taker_at_cross",
    }
    if client_order_id:
        body["client_order_id"] = client_order_id
    _log.info(
        f"place_order V2: ticker={ticker} side={book_side} price={price_str} "
        f"count={count} tif={tif}"
    )
    return await _post("/portfolio/events/orders", body)


async def cancel_order(order_id: str) -> dict:
    return await _delete(f"/portfolio/orders/{order_id}")


async def get_orders(status: str | None = None) -> dict:
    params = {}
    if status:
        params["status"] = status
    return await _get("/portfolio/orders", params=params)


# ── Positions ─────────────────────────────────────────────────────────────────

async def get_positions() -> dict:
    return await _get("/portfolio/positions")


async def get_balance() -> dict:
    return await _get("/portfolio/balance")


async def get_fills(limit: int = 200, cursor: str | None = None) -> dict:
    params: dict = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    return await _get("/portfolio/fills", params=params)
