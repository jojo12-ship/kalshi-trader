"""
Kalshi Autopilot — autonomous trading engine.

Strategies:
  1. Spread Arb   — when yes_ask + no_ask < ARBT_THRESHOLD¢, buy BOTH sides.
                    At expiry, one side pays 100¢ and the other 0¢, so total
                    payout = 100¢ guaranteed. Net profit = 100 - (yes_ask + no_ask).
  2. Drift Trade  — when a market's last_price has drifted >DRIFT_MIN¢ from the
                    current mid, bet that the price will revert to last_price
                    (mean-reversion assumption for thin-book markets).

Risk controls:
  - DAILY_LOSS_LIMIT_USD  : halt all new trades if realized loss exceeds this today
  - MAX_TRADE_USD         : max dollars deployed per single trade
  - DAILY_BUDGET_USD      : total capital to deploy per day (across all trades)
  - CONTRACTS_PER_TRADE   : max contracts per order
  - MIN_SCORE             : minimum scanner score to consider a trade
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import client as kalshi
import crypto_signals as csig
import pnl as pnl_tracker
import positions as pos_mgr
import scanner as mkt_scanner

logger = logging.getLogger("autopilot")


# ── Configurable parameters (can be updated at runtime via API) ───────────────

@dataclass
class AutopilotConfig:
    enabled: bool = False
    scan_interval_sec: int = int(os.getenv("SCAN_INTERVAL_SEC", "15"))   # cooldown between scans
    daily_loss_limit_usd: float = float(os.getenv("DAILY_LOSS_LIMIT_USD", "20.0"))
    daily_budget_usd: float = float(os.getenv("DAILY_BUDGET_USD", "50.0"))
    max_trade_usd: float = float(os.getenv("MAX_TRADE_USD", "5.0"))
    contracts_per_trade: int = int(os.getenv("CONTRACTS_PER_TRADE", "5"))
    min_score: float = float(os.getenv("MIN_SCORE", "5.0"))
    arb_threshold_cents: float = float(os.getenv("ARB_THRESHOLD_CENTS", "98.0"))
    drift_min_cents: float = float(os.getenv("DRIFT_MIN_CENTS", "5.0"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.40"))    # 40% loss
    profit_target_pct: float = float(os.getenv("PROFIT_TARGET_PCT", "0.60"))  # 60% gain
    max_events_to_scan: int = int(os.getenv("MAX_EVENTS_TO_SCAN", "100"))
    scan_category: str | None = None  # None = all categories
    crypto_enabled: bool = False      # enable crypto-market mode + signal bias


config = AutopilotConfig()
_task: asyncio.Task | None = None
_last_result: dict = {}           # most recent completed cycle result
_cycle_running: bool = False      # guard against overlapping manual triggers
_last_crypto_signal: dict = {}    # most recent CryptoSignal serialised for the API

# Track how much we've deployed today
_daily_deployed: dict[str, float] = {}   # date string → dollars deployed

# ── Config persistence ────────────────────────────────────────────────────────
_CONFIG_FILE = Path("config_state.json")
# Fields saved/restored across restarts (subset of AutopilotConfig)
_PERSIST_FIELDS = ("crypto_enabled", "daily_budget_usd", "max_trade_usd", "contracts_per_trade")


def save_config() -> None:
    """Persist selected autopilot config fields to disk."""
    try:
        data = {f: getattr(config, f) for f in _PERSIST_FIELDS}
        _CONFIG_FILE.write_text(json.dumps(data))
    except Exception as exc:
        logger.warning(f"Could not save autopilot config: {exc}")


def load_config() -> None:
    """Restore persisted autopilot config fields from disk (if file exists)."""
    if not _CONFIG_FILE.exists():
        return
    try:
        data = json.loads(_CONFIG_FILE.read_text())
        for f in _PERSIST_FIELDS:
            if f in data:
                setattr(config, f, data[f])
        logger.info(f"Autopilot config restored: {data}")
    except Exception as exc:
        logger.warning(f"Could not restore autopilot config: {exc}")


def _today_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _today_deployed() -> float:
    return _daily_deployed.get(_today_str(), 0.0)


def _add_deployed(amount_usd: float) -> None:
    k = _today_str()
    _daily_deployed[k] = _daily_deployed.get(k, 0.0) + amount_usd


# ── Trade execution ──────────────────────────────────────────────────────────

async def _place_arb(market: dict) -> dict | None:
    """
    Spread arb: buy both yes AND no sides when the gap is below the threshold.
    Both orders are limit orders at the current ask prices.
    """
    ticker = market["ticker"]
    yes_ask = market["yes_ask"]  # cents
    no_ask  = market["no_ask"]   # cents
    gap     = yes_ask + no_ask

    if gap >= config.arb_threshold_cents:
        return None

    yes_price_int = round(yes_ask)
    no_price_int  = round(no_ask)

    # Cost per arb pair in dollars
    cost_per_pair = (yes_price_int + no_price_int) / 100.0
    contracts = min(
        config.contracts_per_trade,
        max(1, int(config.max_trade_usd / cost_per_pair)),
    )
    total_cost = cost_per_pair * contracts

    # Risk budget check
    today_pnl = pnl_tracker.today_realized_pnl()
    if today_pnl <= -config.daily_loss_limit_usd:
        logger.warning("Daily loss limit hit — skipping arb trade")
        return None
    if _today_deployed() + total_cost > config.daily_budget_usd:
        logger.info(f"Daily budget exhausted ({_today_deployed():.2f}/{config.daily_budget_usd:.2f}) — skipping")
        return None

    logger.info(f"ARB {ticker}: gap={gap:.1f}¢  yes@{yes_price_int}¢ + no@{no_price_int}¢  x{contracts}")

    try:
        yes_resp = await kalshi.place_order(
            ticker=ticker,
            side="yes",
            action="buy",
            order_type="limit",
            count=contracts,
            yes_price=yes_price_int,
        )
        no_resp = await kalshi.place_order(
            ticker=ticker,
            side="no",
            action="buy",
            order_type="limit",
            count=contracts,
            yes_price=100 - no_price_int,   # no order uses equivalent yes_price
        )
    except Exception as exc:
        logger.error(f"ARB order failed for {ticker}: {exc}")
        return None

    _add_deployed(total_cost)
    pnl_tracker.record_trade(ticker, "yes", "buy", contracts, yes_price_int, note="arb_yes")
    pnl_tracker.record_trade(ticker, "no",  "buy", contracts, no_price_int,  note="arb_no")

    # Expected profit per pair at expiry (in dollars)
    expected_profit = ((100 - gap) / 100.0) * contracts

    return {
        "strategy": "spread_arb",
        "ticker": ticker,
        "contracts": contracts,
        "yes_price": yes_price_int,
        "no_price": no_price_int,
        "spread_gap": gap,
        "expected_profit_dollars": round(expected_profit, 4),
        "total_cost_dollars": round(total_cost, 4),
        "yes_order": yes_resp,
        "no_order": no_resp,
    }


async def _place_drift_trade(
    market: dict,
    crypto_signal: "csig.CryptoSignal | None" = None,
) -> dict | None:
    """
    Drift trade: if last_price is significantly above current mid, the market
    may be underpriced — buy yes.  If significantly below, buy no.

    When crypto_enabled and a strong signal is present, the signal overrides
    the drift direction for crypto-related markets.
    """
    ticker     = market["ticker"]
    mid        = market["mid"]
    last_price = market["last_price"]
    yes_ask    = market["yes_ask"]
    no_ask     = market["no_ask"]
    drift      = last_price - mid

    if abs(drift) < config.drift_min_cents:
        return None

    # Drift upward → last trade > current mid → market may be underpriced → buy yes
    # Drift downward → last trade < current mid → no side may be underpriced → buy no
    if drift > 0:
        side       = "yes"
        entry_cents = yes_ask
    else:
        side       = "no"
        entry_cents = no_ask

    # ── Crypto signal override ────────────────────────────────────────────────
    # When crypto mode is active and the signal is confident (≥60%), override
    # the drift direction for crypto-related Kalshi markets to align with BTC
    # momentum.  For non-crypto markets the drift direction is kept as-is.
    if (
        config.crypto_enabled
        and crypto_signal
        and not crypto_signal.error
        and crypto_signal.confidence >= 0.60
        and csig.is_crypto_market(market)
    ):
        if crypto_signal.trend == "bullish":
            side        = "yes"
            entry_cents = yes_ask
            logger.info(f"  crypto signal BULLISH → forced YES for {ticker}")
        elif crypto_signal.trend == "bearish":
            side        = "no"
            entry_cents = no_ask
            logger.info(f"  crypto signal BEARISH → forced NO for {ticker}")

    entry_int    = round(entry_cents)
    stop_cents   = round(entry_cents * (1 - config.stop_loss_pct))
    target_cents = min(99, round(entry_cents * (1 + config.profit_target_pct)))
    cost_per_contract = entry_int / 100.0

    contracts = min(
        config.contracts_per_trade,
        max(1, int(config.max_trade_usd / cost_per_contract)),
    )
    total_cost = cost_per_contract * contracts

    today_pnl = pnl_tracker.today_realized_pnl()
    if today_pnl <= -config.daily_loss_limit_usd:
        logger.warning("Daily loss limit hit — skipping drift trade")
        return None
    if _today_deployed() + total_cost > config.daily_budget_usd:
        logger.info(f"Daily budget exhausted — skipping drift trade on {ticker}")
        return None

    # Skip if we already have a managed position on this ticker
    if pos_mgr.get_position(ticker):
        return None

    logger.info(f"DRIFT {ticker}: side={side} entry={entry_int}¢ SL={stop_cents}¢ PT={target_cents}¢")

    try:
        resp = await kalshi.place_order(
            ticker=ticker,
            side=side,
            action="buy",
            order_type="limit",
            count=contracts,
            yes_price=entry_int,  # pass YES price for YES, NO price for NO; client.py converts
        )
    except Exception as exc:
        logger.error(f"Drift order failed for {ticker}: {exc}")
        return None

    _add_deployed(total_cost)
    pos_mgr.add_position(
        ticker=ticker,
        side=side,
        contracts=contracts,
        entry_price=entry_int,
        stop_loss=stop_cents,
        profit_target=target_cents,
    )
    pnl_tracker.record_trade(ticker, side, "buy", contracts, entry_int, note="drift")

    return {
        "strategy": "drift_trade",
        "ticker": ticker,
        "side": side,
        "contracts": contracts,
        "entry_cents": entry_int,
        "stop_loss_cents": stop_cents,
        "profit_target_cents": target_cents,
        "drift_cents": round(drift, 1),
        "total_cost_dollars": round(total_cost, 4),
        "order": resp,
    }


# ── Crypto 15-min market trading ─────────────────────────────────────────────

# Scan window when hunting for 15-min crypto markets: look 90 min ahead so we
# always have at least one full 15-min window queued up ahead of us.
_CRYPTO_SCAN_WINDOW_SEC = 90 * 60

# Minimum seconds-to-close before we'll enter a crypto market.
# Buying with <5 min left is basically flipping a coin at the last second.
_CRYPTO_MIN_TTL_SEC = 5 * 60


async def _fetch_near_term_crypto_markets() -> list[dict]:
    """
    Return open Kalshi 15-min and hourly crypto directional markets closing
    within the next 90 minutes.

    Fetches by series_ticker (discovered via GET /series — confirmed as
    freq=fifteen_min / hourly) instead of scanning all markets and guessing
    ticker prefixes.  This correctly finds KXHYPE15M, KXDOGE15M, KXBNB, etc.
    """
    now_ts  = _time.time()
    deadline = now_ts + _CRYPTO_SCAN_WINDOW_SEC

    seen: dict[str, dict] = {}  # ticker → market (dedup across series)

    # Only 15-min "Up or Down" series — the hourly series are price-strike markets
    # ("Will BTC be above $X?"), not directional Up/Down, so they're not suited
    # for BTC-momentum-based trading.  The 5–70¢ filter handles any strays.
    for series in csig.CRYPTO_SERIES_15M:
        try:
            data = await kalshi.get_markets(
                series_ticker=series,
                status="open",
                limit=50,
            )
            for m in data.get("markets", []):
                close_str = m.get("close_time", "")
                try:
                    ct = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    if ct.timestamp() > deadline:
                        continue
                except Exception:
                    pass  # keep and let TTL guard in _trade_crypto_markets handle it
                ticker = m.get("ticker", "")
                if ticker:
                    seen[ticker] = m
        except Exception as exc:
            logger.warning(f"Crypto series {series} fetch failed: {exc}")

    markets = list(seen.values())
    logger.info(
        f"Near-term crypto market scan: {len(csig.CRYPTO_SERIES_15M)} 15-min series "
        f"→ {len(markets)} market(s) closing within {_CRYPTO_SCAN_WINDOW_SEC // 60} min"
    )
    return markets


def _count_open_crypto_positions() -> int:
    """Count currently managed positions that are 15-min crypto markets."""
    return sum(
        1 for p in pos_mgr.get_all_positions()
        if "15M" in p.get("ticker", "").upper()
    )


async def _trade_crypto_markets(crypto_sig: "csig.CryptoSignal") -> list[dict]:
    """
    Dedicated 15-min crypto market trader.

    Logic:
      - Skip if trend is neutral or confidence < CRYPTO_CONF_MIN.
      - Skip if already at MAX_CRYPTO_POSITIONS open at once (cap concentration).
      - Fetch all open Kalshi crypto markets closing within 90 min.
      - Bullish → buy YES; Bearish → buy NO.
      - Skip markets with entry price outside 5–60¢ (terrible risk/reward above 60¢).
      - Skip markets with < 5 min to close (too late to enter).
      - Respect daily budget and position deduplication.
    """
    CRYPTO_CONF_MIN = 0.55
    MAX_CRYPTO_POSITIONS = 2   # never hold more than 2 open crypto slots at once
    trades: list[dict] = []

    if crypto_sig.error:
        logger.info(f"Crypto pass skipped: signal error — {crypto_sig.error}")
        return trades

    if crypto_sig.trend == "neutral" or crypto_sig.confidence < CRYPTO_CONF_MIN:
        logger.info(
            f"Crypto pass skipped: trend={crypto_sig.trend} "
            f"conf={crypto_sig.confidence:.2f} (need bullish/bearish ≥{CRYPTO_CONF_MIN})"
        )
        return trades

    open_crypto = _count_open_crypto_positions()
    if open_crypto >= MAX_CRYPTO_POSITIONS:
        logger.info(
            f"Crypto pass skipped: {open_crypto}/{MAX_CRYPTO_POSITIONS} slots already open"
        )
        return trades

    markets = await _fetch_near_term_crypto_markets()
    if not markets:
        logger.info("Crypto pass: no near-term Kalshi crypto markets found")
        return trades

    side = "yes" if crypto_sig.trend == "bullish" else "no"
    logger.info(
        f"Crypto pass: {crypto_sig.trend.upper()} conf={crypto_sig.confidence:.2f} "
        f"BTC=${crypto_sig.btc_price:,.0f} — {len(markets)} market(s) found → {side.upper()}"
    )

    now_ts = _time.time()
    placed = 0

    for mkt in markets:
        ticker = mkt.get("ticker", "")

        # Guard: skip markets closing too soon
        close_str = mkt.get("close_time", "")
        try:
            ct = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
            ttl = ct.timestamp() - now_ts
            if ttl < _CRYPTO_MIN_TTL_SEC:
                logger.debug(f"  {ticker}: closes in {ttl:.0f}s — too close, skip")
                continue
        except Exception:
            pass  # if we can't parse close_time, proceed cautiously

        yes_ask  = round(float(mkt.get("yes_ask_dollars") or 0) * 100)
        no_ask   = round(float(mkt.get("no_ask_dollars")  or 0) * 100)
        entry_cents = yes_ask if side == "yes" else no_ask

        # Price ceiling: only enter where risk/reward is reasonable (skip >60¢ — priced in)
        if entry_cents < 5 or entry_cents >= 60:
            logger.debug(f"  {ticker}: {side} ask={entry_cents}¢ — outside 5–70¢, skip")
            continue

        # Don't double-enter existing managed positions
        if pos_mgr.get_position(ticker):
            continue

        # Daily loss / budget gates
        if pnl_tracker.today_realized_pnl() <= -config.daily_loss_limit_usd:
            logger.warning("Daily loss limit hit — stopping crypto trades")
            break

        cost_per  = entry_cents / 100.0
        contracts = min(
            config.contracts_per_trade,
            max(1, int(config.max_trade_usd / cost_per)),
        )
        total_cost = cost_per * contracts

        if _today_deployed() + total_cost > config.daily_budget_usd:
            logger.info("Daily budget exhausted — stopping crypto trades")
            break

        stop_cents   = round(entry_cents * (1 - config.stop_loss_pct))
        target_cents = min(99, round(entry_cents * (1 + config.profit_target_pct)))

        logger.info(
            f"  CRYPTO {ticker}: {crypto_sig.trend.upper()} → {side.upper()} "
            f"entry={entry_cents}¢ SL={stop_cents}¢ PT={target_cents}¢ "
            f"x{contracts} (conf={crypto_sig.confidence:.2f})"
        )

        try:
            resp = await kalshi.place_order(
                ticker=ticker,
                side=side,
                action="buy",
                order_type="limit",
                count=contracts,
                yes_price=entry_cents,
            )
        except Exception as exc:
            logger.error(f"  Crypto order failed for {ticker}: {exc}")
            continue

        _add_deployed(total_cost)
        pos_mgr.add_position(
            ticker=ticker,
            side=side,
            contracts=contracts,
            entry_price=entry_cents,
            stop_loss=stop_cents,
            profit_target=target_cents,
        )
        pnl_tracker.record_trade(
            ticker, side, "buy", contracts, entry_cents, note="crypto_15m"
        )

        trades.append({
            "strategy":             "crypto_15m",
            "ticker":               ticker,
            "side":                 side,
            "trend":                crypto_sig.trend,
            "confidence":           crypto_sig.confidence,
            "btc_price":            crypto_sig.btc_price,
            "contracts":            contracts,
            "entry_cents":          entry_cents,
            "stop_loss_cents":      stop_cents,
            "profit_target_cents":  target_cents,
            "total_cost_dollars":   round(total_cost, 4),
            "order":                resp,
        })
        placed += 1
        await asyncio.sleep(0.3)

        if placed >= 5:  # cap per cycle — don't over-concentrate in crypto
            break

    logger.info(f"Crypto pass done: {placed} trade(s) placed")
    return trades


# ── Main scan-and-trade loop ─────────────────────────────────────────────────

async def _run_cycle() -> dict:
    """Single autopilot cycle: scan → filter → trade."""
    today_pnl = pnl_tracker.today_realized_pnl()
    if today_pnl <= -config.daily_loss_limit_usd:
        msg = f"Daily loss limit of ${config.daily_loss_limit_usd:.2f} reached. Autopilot halted for today."
        logger.warning(msg)
        return {"halted": True, "reason": msg, "today_pnl": today_pnl}

    if _today_deployed() >= config.daily_budget_usd:
        msg = f"Daily budget of ${config.daily_budget_usd:.2f} fully deployed."
        logger.info(msg)
        return {"halted": False, "reason": msg, "deployed": _today_deployed()}

    # ── Crypto signal (non-blocking) ──────────────────────────────────────────
    global _last_crypto_signal
    crypto_sig: csig.CryptoSignal | None = None
    if config.crypto_enabled:
        try:
            crypto_sig = await csig.get_crypto_signal()
            _last_crypto_signal = crypto_sig.to_dict()
            logger.info(
                f"Crypto signal: {crypto_sig.trend} "
                f"conf={crypto_sig.confidence:.2f} "
                f"BTC=${crypto_sig.btc_price:,.0f} "
                f"5m={crypto_sig.btc_change_5m:+.3f}%"
            )
        except Exception as exc:
            logger.warning(f"Crypto signal fetch failed: {exc}")

    logger.info("Autopilot cycle: scanning markets…")
    try:
        candidates, total_scanned = await mkt_scanner.scan_markets(
            max_events=config.max_events_to_scan,
            min_score=config.min_score,
            category=config.scan_category,
        )
    except Exception as exc:
        logger.error(f"Scan failed: {exc}")
        return {"error": str(exc)}

    trades_placed = []
    arb_candidates  = [m for m in candidates if m["spread_gap"] < config.arb_threshold_cents]
    drift_candidates = [
        m for m in candidates
        if abs(m["last_price"] - m["mid"]) >= config.drift_min_cents
        and m["spread_gap"] >= config.arb_threshold_cents
    ]

    # In crypto mode, separate crypto markets and prioritise them
    crypto_drift: list[dict] = []
    normal_drift = drift_candidates
    if config.crypto_enabled and crypto_sig:
        crypto_drift = [m for m in drift_candidates if csig.is_crypto_market(m)]
        normal_drift = [m for m in drift_candidates if not csig.is_crypto_market(m)]
        if crypto_drift:
            logger.info(f"Crypto mode: {len(crypto_drift)} crypto market candidate(s)")

    logger.info(
        f"Scanned {total_scanned} markets → {len(candidates)} signals "
        f"({len(arb_candidates)} arb, {len(drift_candidates)} drift)"
    )

    # Execute arb trades first (highest priority — near-guaranteed profit)
    for market in arb_candidates[:3]:
        result = await _place_arb(market)
        if result:
            trades_placed.append(result)
        await asyncio.sleep(0.5)

    # Dedicated crypto 15-min market pass (runs when crypto mode is on)
    # This is separate from drift — 15-min markets rarely show drift signals
    # because they're efficiently priced. We trade direction from the signal.
    if config.crypto_enabled and crypto_sig:
        crypto_15m_trades = await _trade_crypto_markets(crypto_sig)
        trades_placed.extend(crypto_15m_trades)

    # Crypto-market drift trades (signal-guided direction)
    for market in crypto_drift[:2]:
        result = await _place_drift_trade(market, crypto_signal=crypto_sig)
        if result:
            trades_placed.append(result)
        await asyncio.sleep(0.5)

    # Normal drift trades (momentum / spread)
    for market in normal_drift[:3]:
        result = await _place_drift_trade(market)
        if result:
            trades_placed.append(result)
        await asyncio.sleep(0.5)

    # Run position monitor
    monitor_actions = await pos_mgr.monitor_once()

    return {
        "scanned": total_scanned,
        "signals": len(candidates),
        "trades_placed": len(trades_placed),
        "trades": trades_placed,
        "monitor_actions": monitor_actions,
        "today_deployed_dollars": round(_today_deployed(), 2),
        "today_pnl_dollars": round(pnl_tracker.today_realized_pnl(), 4),
        "crypto_signal": _last_crypto_signal if config.crypto_enabled else None,
    }


async def _run_cycle_tracked() -> dict:
    global _cycle_running, _last_result
    _cycle_running = True
    try:
        result = await _run_cycle()
        _last_result = result
        return result
    finally:
        _cycle_running = False


async def _loop():
    logger.info("Autopilot loop started — continuous scanning mode")
    while True:
        if config.enabled and not _cycle_running:
            try:
                result = await _run_cycle_tracked()
                trades = result.get("trades_placed", 0)
                logger.info(
                    f"Cycle done: {result.get('scanned',0)} markets, "
                    f"{result.get('signals',0)} signals, {trades} trade(s) placed"
                )
            except Exception as exc:
                logger.exception(f"Cycle error: {exc}")
            await asyncio.sleep(max(5, config.scan_interval_sec))
        else:
            await asyncio.sleep(2)


def trigger_cycle_background() -> bool:
    """Fire a cycle in the background if one isn't already running. Returns True if triggered."""
    if _cycle_running:
        return False
    asyncio.ensure_future(_run_cycle_tracked())
    return True


def start_task(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
    global _task
    if _task and not _task.done():
        return _task
    _task = asyncio.ensure_future(_loop())
    return _task


def get_status() -> dict:
    return {
        "enabled": config.enabled,
        "scan_interval_sec": config.scan_interval_sec,
        "daily_loss_limit_usd": config.daily_loss_limit_usd,
        "daily_budget_usd": config.daily_budget_usd,
        "max_trade_usd": config.max_trade_usd,
        "contracts_per_trade": config.contracts_per_trade,
        "min_score": config.min_score,
        "arb_threshold_cents": config.arb_threshold_cents,
        "drift_min_cents": config.drift_min_cents,
        "stop_loss_pct": config.stop_loss_pct,
        "profit_target_pct": config.profit_target_pct,
        "max_events_to_scan": config.max_events_to_scan,
        "scan_category": config.scan_category,
        "crypto_enabled": config.crypto_enabled,
        "today_deployed_dollars": round(_today_deployed(), 2),
        "today_realized_pnl": pnl_tracker.today_realized_pnl(),
        "pnl_summary": pnl_tracker.summary(),
        "crypto_signal": _last_crypto_signal if config.crypto_enabled else None,
    }
