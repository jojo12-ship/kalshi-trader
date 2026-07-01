"""
Position manager — tracks open positions and enforces stop-loss / profit targets.

Positions are persisted to `positions_state.json` alongside this file so that
a service restart does not cause re-entry into already-held markets.

Stop-loss and profit targets are expressed as yes-price in cents (1–99).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import client as kalshi
import pnl as pnl_tracker

logger = logging.getLogger("positions")

_STATE_FILE = os.path.join(os.path.dirname(__file__), "positions_state.json")


@dataclass
class ManagedPosition:
    ticker: str
    side: str          # "yes" or "no"
    contracts: int
    entry_price: int   # cents
    stop_loss: int     # yes-price in cents — exit if yes falls to/below this
    profit_target: int # yes-price in cents — exit if yes rises to/above this
    opened_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "open"  # "open" | "closed" | "error"
    exit_price: int | None = None
    exit_reason: str | None = None
    exit_at: datetime | None = None


# In-memory store: ticker -> ManagedPosition
_positions: dict[str, ManagedPosition] = {}

# Unrealized P&L cache — updated by monitor_once(), cleared on close
_unrealized_pnl: dict[str, float] = {}   # ticker → dollars


def get_unrealized_pnl() -> tuple[float, dict[str, float]]:
    """Return (total_unrealized_dollars, per_ticker_dict). Thread-safe snapshot."""
    by_ticker = dict(_unrealized_pnl)
    return round(sum(by_ticker.values()), 4), by_ticker


# ── Persistence ──────────────────────────────────────────────────────────────

def _save() -> None:
    """Write all managed positions to disk atomically."""
    try:
        data = []
        for p in _positions.values():
            d = {
                "ticker": p.ticker,
                "side": p.side,
                "contracts": p.contracts,
                "entry_price": p.entry_price,
                "stop_loss": p.stop_loss,
                "profit_target": p.profit_target,
                "status": p.status,
                "exit_price": p.exit_price,
                "exit_reason": p.exit_reason,
                "opened_at": p.opened_at.isoformat(),
                "exit_at": p.exit_at.isoformat() if p.exit_at else None,
            }
            data.append(d)
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    except Exception as exc:
        logger.warning(f"Failed to save position state: {exc}")


def load_from_disk() -> int:
    """Load persisted positions on startup. Returns number loaded."""
    if not os.path.exists(_STATE_FILE):
        return 0
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)
        loaded = 0
        for d in data:
            # Only reload open positions — closed ones are historical
            if d.get("status") not in ("open",):
                continue
            pos = ManagedPosition(
                ticker=d["ticker"],
                side=d["side"],
                contracts=d["contracts"],
                entry_price=d["entry_price"],
                stop_loss=d["stop_loss"],
                profit_target=d["profit_target"],
                opened_at=datetime.fromisoformat(d["opened_at"]),
                status=d["status"],
                exit_price=d.get("exit_price"),
                exit_reason=d.get("exit_reason"),
                exit_at=datetime.fromisoformat(d["exit_at"]) if d.get("exit_at") else None,
            )
            _positions[pos.ticker] = pos
            loaded += 1
        if loaded:
            logger.info(f"Restored {loaded} managed position(s) from disk")
        return loaded
    except Exception as exc:
        logger.warning(f"Failed to load position state: {exc}")
        return 0


# ── CRUD ─────────────────────────────────────────────────────────────────────

def add_position(
    ticker: str,
    side: str,
    contracts: int,
    entry_price: int,
    stop_loss: int,
    profit_target: int,
) -> ManagedPosition:
    pos = ManagedPosition(
        ticker=ticker,
        side=side,
        contracts=contracts,
        entry_price=entry_price,
        stop_loss=stop_loss,
        profit_target=profit_target,
    )
    _positions[ticker] = pos
    _save()
    logger.info(f"Tracking position {ticker}: entry={entry_price}¢ SL={stop_loss}¢ PT={profit_target}¢")
    return pos


def get_all_positions() -> list[dict]:
    return [_position_to_dict(p) for p in _positions.values()]


def get_position(ticker: str) -> ManagedPosition | None:
    return _positions.get(ticker)


def remove_position(ticker: str) -> bool:
    removed = _positions.pop(ticker, None) is not None
    if removed:
        _save()
    return removed


def _position_to_dict(p: ManagedPosition) -> dict:
    return {
        "ticker": p.ticker,
        "side": p.side,
        "contracts": p.contracts,
        "entry_price": p.entry_price,
        "stop_loss": p.stop_loss,
        "profit_target": p.profit_target,
        "status": p.status,
        "exit_price": p.exit_price,
        "exit_reason": p.exit_reason,
        "opened_at": p.opened_at.isoformat(),
        "exit_at": p.exit_at.isoformat() if p.exit_at else None,
    }


# ── Position monitor ──────────────────────────────────────────────────────────

def _calc_close_pnl(pos: ManagedPosition, exit_yes_price: int) -> float:
    """Realized P&L in dollars when closing at exit_yes_price (yes-side cents)."""
    if pos.side == "yes":
        return (exit_yes_price - pos.entry_price) / 100.0 * pos.contracts
    else:
        # entry_price is the no_ask paid; NO value ≈ 100 - current_yes
        no_exit = 100 - exit_yes_price
        return (no_exit - pos.entry_price) / 100.0 * pos.contracts


def _record_close(pos: ManagedPosition, exit_yes_price: int, reason: str) -> float:
    """Mark position closed, update P&L tracker, save. Returns realized P&L."""
    pnl = _calc_close_pnl(pos, exit_yes_price)
    pos.status     = "closed"
    pos.exit_price = exit_yes_price
    pos.exit_reason = reason
    pos.exit_at    = datetime.utcnow()
    _unrealized_pnl.pop(pos.ticker, None)
    _save()
    pnl_tracker.record_trade(
        pos.ticker, pos.side, "sell", pos.contracts,
        exit_yes_price, pnl_dollars=round(pnl, 4), note=reason,
    )
    logger.info(f"Closed {pos.ticker} side={pos.side} @ {exit_yes_price}¢ — {reason} → P&L ${pnl:+.2f}")
    return pnl


async def _close_position(pos: ManagedPosition, reason: str, current_yes_price: int) -> None:
    """Place a market sell order to exit the position, then record P&L."""
    try:
        resp = await kalshi.place_order(
            ticker=pos.ticker,
            side=pos.side,
            action="sell",
            order_type="market",
            count=pos.contracts,
        )
        _record_close(pos, current_yes_price, reason)
        logger.debug(f"Sell order for {pos.ticker}: {resp}")
    except Exception as exc:
        pos.status = "error"
        _save()
        logger.error(f"Failed to close {pos.ticker}: {exc}")


async def monitor_once() -> list[dict]:
    """
    Check current prices for all open managed positions.
    Closes any that have breached their stop-loss or profit target.
    Returns a list of actions taken this cycle.
    """
    open_positions = [p for p in _positions.values() if p.status == "open"]
    if not open_positions:
        return []

    # Fetch live market data for all open positions concurrently
    results = await asyncio.gather(
        *[kalshi.get_market(p.ticker) for p in open_positions],
        return_exceptions=True,
    )

    actions = []
    for pos, result in zip(open_positions, results):
        if isinstance(result, Exception):
            logger.warning(f"Failed to fetch price for {pos.ticker}: {result}")
            continue

        market = result.get("market", result)

        # ── Settlement detection ───────────────────────────────────────────────
        mkt_status = market.get("status", "")
        if mkt_status == "finalized":
            mkt_result = market.get("result", "")   # "yes" or "no"
            if mkt_result in ("yes", "no"):
                won = pos.side == mkt_result
                # result=="yes" → YES price settles at 100¢; result=="no" → YES settles at 0¢
                settlement_yes = 100 if mkt_result == "yes" else 0
                pnl = _record_close(pos, settlement_yes, f"settled_{'win' if won else 'loss'}")
                actions.append({
                    "ticker": pos.ticker,
                    "trigger": f"settled_{mkt_result}",
                    "side": pos.side,
                    "won": won,
                    "pnl": round(pnl, 4),
                })
            else:
                # Finalized but no result yet — skip for now
                logger.debug(f"{pos.ticker} finalized but no result field yet")
            continue   # never attempt SL/PT on a finalized market

        def _c(val: object) -> float:
            try:
                return float(val) * 100
            except (TypeError, ValueError):
                return 0.0

        yes_bid_c = _c(market.get("yes_bid_dollars"))
        yes_ask_c = _c(market.get("yes_ask_dollars"))
        last_c    = _c(market.get("last_price_dollars"))

        # Use mid of bid/ask when both sides are valid; fall back to last_price
        if yes_bid_c > 0 and yes_ask_c > 0 and yes_ask_c < 100:
            current_yes = (yes_bid_c + yes_ask_c) / 2.0
        elif last_c > 0:
            current_yes = last_c
        else:
            logger.debug(f"No valid price for {pos.ticker} — skipping SL/PT check")
            continue

        # Cache unrealized P&L for this position
        if pos.side == "yes":
            upnl = (current_yes - pos.entry_price) / 100.0 * pos.contracts
        else:
            # NO position: profit when YES price falls below our no_ask entry
            no_current = 100 - current_yes
            upnl = (no_current - pos.entry_price) / 100.0 * pos.contracts
        _unrealized_pnl[pos.ticker] = round(upnl, 4)

        # Grace period: don't trigger SL/PT in the first 5 minutes after entry
        age_seconds = (datetime.utcnow() - pos.opened_at).total_seconds()
        if age_seconds < 300:
            logger.debug(f"{pos.ticker} grace period ({age_seconds:.0f}s < 300s) — skipping")
            continue

        # SL/PT logic: thresholds are in yes-price space
        # For YES positions: profit when YES rises; for NO positions: profit when YES falls
        action = None
        if pos.side == "yes":
            if current_yes <= pos.stop_loss:
                action = {"ticker": pos.ticker, "trigger": "stop_loss", "price": round(current_yes)}
                await _close_position(pos, "stop_loss", round(current_yes))
            elif current_yes >= pos.profit_target:
                action = {"ticker": pos.ticker, "trigger": "profit_target", "price": round(current_yes)}
                await _close_position(pos, "profit_target", round(current_yes))
        else:
            # NO position: stop loss when YES rises; profit target when YES falls
            # Reinterpret thresholds in no-price space
            no_current_i = round(100 - current_yes)
            if no_current_i <= pos.stop_loss:
                action = {"ticker": pos.ticker, "trigger": "stop_loss", "price": round(current_yes)}
                await _close_position(pos, "stop_loss", round(current_yes))
            elif no_current_i >= pos.profit_target:
                action = {"ticker": pos.ticker, "trigger": "profit_target", "price": round(current_yes)}
                await _close_position(pos, "profit_target", round(current_yes))

        if action:
            actions.append(action)

    return actions
