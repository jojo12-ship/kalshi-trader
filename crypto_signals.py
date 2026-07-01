"""
Crypto signal engine — live BTC/ETH candle data + news sentiment.

Sources (free, no API key required):
  Binance REST API  : 5-minute OHLCV candles for BTC and ETH
  CryptoCompare API : recent headlines (basic tier, unauthenticated)

Usage:
  signal = await get_crypto_signal()
  if signal.trend == "bullish" and signal.confidence > 0.6:
      # prefer YES on bullish crypto markets
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("crypto")

BINANCE_BASE = "https://api.binance.us/api/v3"  # .com is geo-blocked on this server
CC_NEWS_URL  = (
    "https://min-api.cryptocompare.com/data/v2/news/"
    "?lang=EN&categories=BTC,ETH&sortOrder=popular"
)

_BULLISH_WORDS = {
    "rally","surge","breakout","bull","bullish","pump","ath","record","high",
    "adoption","approved","approval","buy","positive","gain","rise","soar",
    "accumulate","institutional","etf","inflow","support","recover","rebound",
}
_BEARISH_WORDS = {
    "crash","dump","drop","fall","bear","bearish","plunge","decline","sell",
    "selloff","hack","ban","regulatory","warning","risk","low","loss","short",
    "correction","capitulation","liquidation","outflow","resistance","breakdown",
}

# Kalshi crypto market detection keywords (used for drift-trade classification).
# Deliberately narrow to avoid false positives — "ETH" appears in "NETHERLANDS",
# "SOL" in "CONSOLATION", etc.
_CRYPTO_KEYWORDS = {
    "BTC", "BITCOIN", "ETHEREUM", "SOLANA", "COINBASE", "KXBTC", "KXETH", "KXSOL",
}

# ── Kalshi series tickers for the dedicated crypto trading pass ───────────────
# These were discovered via GET /series and confirmed as freq=fifteen_min / hourly.
# Fetch markets using GET /markets?series_ticker=<series> rather than scanning
# all markets and hoping for prefix matches.

# 15-minute "Up or Down" crypto markets (the primary target)
CRYPTO_SERIES_15M: list[str] = [
    "KXHYPE15M",   # HYPE (Hyperliquid) 15 min
    "KXETH15M",    # Ethereum 15 min
    "KXSOL15M",    # Solana 15 min
    "KXDOGE15M",   # Dogecoin 15 min
    "KXNEAR15M",   # NEAR Protocol 15 min
]

# Hourly directional crypto markets (secondary — enter if signal is strong)
CRYPTO_SERIES_HOURLY: list[str] = [
    "KXBTC",    # Bitcoin range (hourly)
    "KXBNB",    # BNB (hourly)
    "KXBNBD",   # BNB Directional (hourly)
    "KXHYPE",   # HYPE (hourly)
    "KXETHD",   # Ethereum Directional (hourly)
    "KXDOGED",  # Dogecoin Directional (hourly)
    "KXTOND",   # TON Directional (hourly)
    "KXXRP",    # XRP Range (hourly)
]

ALL_CRYPTO_SERIES: list[str] = CRYPTO_SERIES_15M + CRYPTO_SERIES_HOURLY

# Legacy prefix list kept for is_crypto_market() fallback
CRYPTO_TICKER_PREFIXES = tuple(ALL_CRYPTO_SERIES)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CryptoSignal:
    btc_price: float = 0.0
    eth_price: float = 0.0
    btc_change_5m: float = 0.0   # % change over last 5 min candle
    btc_change_1h: float = 0.0   # % change over last 12 × 5m candles
    trend: str = "neutral"       # "bullish" | "bearish" | "neutral"
    confidence: float = 0.0      # 0.0–1.0 blended signal strength
    news_sentiment: float = 0.0  # -1.0 (very bearish) … +1.0 (very bullish)
    news_headlines: list[str] = field(default_factory=list)
    candles_5m: list[list] = field(default_factory=list)  # raw Binance klines
    fetched_at: float = field(default_factory=time.time)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "btc_price":       self.btc_price,
            "eth_price":       self.eth_price,
            "btc_change_5m":   self.btc_change_5m,
            "btc_change_1h":   self.btc_change_1h,
            "trend":           self.trend,
            "confidence":      self.confidence,
            "news_sentiment":  self.news_sentiment,
            "news_headlines":  self.news_headlines,
            # send only [ts, open, high, low, close] for the last 16 candles
            "candles_5m": [
                [c[0], c[1], c[2], c[3], c[4]]
                for c in self.candles_5m[-16:]
            ],
            "fetched_at": self.fetched_at,
            "error":      self.error,
        }


# ── Fetchers ─────────────────────────────────────────────────────────────────

async def _fetch_candles(
    client: httpx.AsyncClient,
    symbol: str,
    interval: str = "5m",
    limit: int = 24,
) -> list[list]:
    try:
        r = await client.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=8.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning(f"Binance {symbol} candles: {exc}")
        return []


async def _fetch_news(client: httpx.AsyncClient) -> list[dict]:
    try:
        r = await client.get(CC_NEWS_URL, timeout=8.0)
        r.raise_for_status()
        return r.json().get("Data", [])[:20]
    except Exception as exc:
        logger.warning(f"CryptoCompare news: {exc}")
        return []


# ── Analysis ─────────────────────────────────────────────────────────────────

def _compute_trend(
    candles: list[list],
) -> tuple[str, float, float, float]:
    """Return (trend, confidence, change_5m_pct, change_1h_pct)."""
    if len(candles) < 20:
        return "neutral", 0.2, 0.0, 0.0

    closes  = [float(c[4]) for c in candles]
    current = closes[-1]
    prev_5m = closes[-2]
    prev_1h = closes[-12]

    change_5m = (current - prev_5m) / prev_5m * 100 if prev_5m else 0.0
    change_1h = (current - prev_1h) / prev_1h * 100 if prev_1h else 0.0

    ma5  = sum(closes[-5:])  / 5
    ma20 = sum(closes[-20:]) / 20

    bull = sum([
        change_5m >  0.08,
        change_1h >  0.25,
        ma5 > ma20,
        current > ma5,
    ])
    bear = sum([
        change_5m < -0.08,
        change_1h < -0.25,
        ma5 < ma20,
        current < ma5,
    ])

    if bull >= 3:
        return "bullish", round(bull / 4.0, 3), change_5m, change_1h
    if bear >= 3:
        return "bearish", round(bear / 4.0, 3), change_5m, change_1h
    return "neutral", 0.3, change_5m, change_1h


def _news_sentiment(articles: list[dict]) -> tuple[float, list[str]]:
    if not articles:
        return 0.0, []
    headlines = [a.get("title", "") for a in articles[:10]]
    score = 0
    for h in headlines:
        words = set(h.lower().split())
        score += len(words & _BULLISH_WORDS)
        score -= len(words & _BEARISH_WORDS)
    normalized = max(-1.0, min(1.0, score / max(len(headlines), 1)))
    return round(normalized, 3), headlines[:6]


# ── Public API ────────────────────────────────────────────────────────────────

async def get_crypto_signal() -> CryptoSignal:
    """Fetch BTC/ETH candles + crypto news; return a blended CryptoSignal."""
    async with httpx.AsyncClient() as client:
        btc_candles, eth_candles, articles = await asyncio.gather(
            _fetch_candles(client, "BTCUSDT", "5m", 24),
            _fetch_candles(client, "ETHUSDT", "5m",  4),
            _fetch_news(client),
        )

    if not btc_candles:
        return CryptoSignal(error="Binance unavailable — no candle data")

    trend, confidence, ch5m, ch1h = _compute_trend(btc_candles)
    news_sent, headlines = _news_sentiment(articles)

    # News confirms or contradicts price momentum
    confirms  = (news_sent >  0.1 and trend == "bullish") or (news_sent < -0.1 and trend == "bearish")
    conflicts = (news_sent >  0.1 and trend == "bearish") or (news_sent < -0.1 and trend == "bullish")
    if confirms:
        confidence = min(1.0, confidence + 0.12)
    elif conflicts:
        confidence = max(0.0, confidence - 0.08)

    return CryptoSignal(
        btc_price      = round(float(btc_candles[-1][4]), 2),
        eth_price      = round(float(eth_candles[-1][4]), 2) if eth_candles else 0.0,
        btc_change_5m  = round(ch5m, 4),
        btc_change_1h  = round(ch1h, 4),
        trend          = trend,
        confidence     = round(confidence, 3),
        news_sentiment = news_sent,
        news_headlines = headlines,
        candles_5m     = btc_candles[-16:],
    )


def is_crypto_market(market: dict) -> bool:
    """True if a Kalshi market belongs to a known crypto series or has crypto keywords."""
    event  = (market.get("event_ticker") or "").upper()
    ticker = (market.get("ticker")       or "").upper()
    title  = (market.get("title")        or "").upper()
    # Check whether the market's event_ticker or ticker starts with any known series
    if any(event.startswith(s.upper()) or ticker.startswith(s.upper()) for s in ALL_CRYPTO_SERIES):
        return True
    return any(k in ticker or k in title for k in _CRYPTO_KEYWORDS)
