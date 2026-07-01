"""
Daily P&L tracker.

Tracks realized and unrealized P&L, trade history, and enforces daily limits.
All values in dollars (float).

State is persisted to pnl_state.json so restarts don't wipe history.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger("pnl")

_STATE_FILE = os.path.join(os.path.dirname(__file__), "pnl_state.json")


@dataclass
class TradeRecord:
    ticker: str
    side: str            # "yes" or "no"
    action: str          # "buy" | "sell" | "settle"
    contracts: int
    price_cents: float
    filled_at: datetime = field(default_factory=datetime.utcnow)
    pnl_dollars: float = 0.0   # realized P&L when closed
    note: str = ""


_lock = threading.Lock()
_trades: list[TradeRecord] = []
_daily_realized: dict[date, float] = {}   # date → realized $ P&L
_session_start_balance: float | None = None


# ── Persistence ───────────────────────────────────────────────────────────────

def _save() -> None:
    try:
        data = {
            "daily_realized": {str(d): v for d, v in _daily_realized.items()},
            "session_start_balance": _session_start_balance,
            "trades": [
                {
                    "ticker": t.ticker,
                    "side": t.side,
                    "action": t.action,
                    "contracts": t.contracts,
                    "price_cents": t.price_cents,
                    "filled_at": t.filled_at.isoformat(),
                    "pnl_dollars": t.pnl_dollars,
                    "note": t.note,
                }
                for t in _trades
            ],
        }
        tmp = _STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, _STATE_FILE)
    except Exception as exc:
        logger.warning(f"Failed to save P&L state: {exc}")


def load_from_disk() -> None:
    """Load persisted P&L state on startup."""
    global _session_start_balance
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE) as f:
            data = json.load(f)

        with _lock:
            _daily_realized.clear()
            for ds, v in data.get("daily_realized", {}).items():
                _daily_realized[date.fromisoformat(ds)] = v

            _trades.clear()
            for t in data.get("trades", []):
                _trades.append(TradeRecord(
                    ticker=t["ticker"],
                    side=t["side"],
                    action=t["action"],
                    contracts=t["contracts"],
                    price_cents=t["price_cents"],
                    filled_at=datetime.fromisoformat(t["filled_at"]),
                    pnl_dollars=t["pnl_dollars"],
                    note=t["note"],
                ))

            _session_start_balance = data.get("session_start_balance")

        loaded_trades = len(data.get("trades", []))
        today_pnl = _daily_realized.get(date.today(), 0.0)
        logger.info(
            f"Loaded P&L state: {loaded_trades} trades, today=${today_pnl:+.2f}"
        )
    except Exception as exc:
        logger.warning(f"Failed to load P&L state: {exc}")


# ── Public API ────────────────────────────────────────────────────────────────

def set_session_start_balance(balance_dollars: float) -> None:
    global _session_start_balance
    with _lock:
        if _session_start_balance is None:
            _session_start_balance = balance_dollars
    _save()


def record_trade(
    ticker: str,
    side: str,
    action: str,
    contracts: int,
    price_cents: float,
    pnl_dollars: float = 0.0,
    note: str = "",
) -> TradeRecord:
    rec = TradeRecord(
        ticker=ticker,
        side=side,
        action=action,
        contracts=contracts,
        price_cents=price_cents,
        pnl_dollars=pnl_dollars,
        note=note,
    )
    with _lock:
        _trades.append(rec)
        today = date.today()
        _daily_realized[today] = _daily_realized.get(today, 0.0) + pnl_dollars
    _save()
    return rec


def today_realized_pnl() -> float:
    with _lock:
        return _daily_realized.get(date.today(), 0.0)


def all_time_realized_pnl() -> float:
    with _lock:
        return sum(_daily_realized.values())


def today_trades() -> list[TradeRecord]:
    today = date.today()
    with _lock:
        return [t for t in _trades if t.filled_at.date() == today]


def summary() -> dict:
    with _lock:
        today = date.today()
        today_pnl = _daily_realized.get(today, 0.0)
        all_time = sum(_daily_realized.values())
        today_count = sum(1 for t in _trades if t.filled_at.date() == today)
        return {
            "today_realized_pnl_dollars": round(today_pnl, 4),
            "all_time_realized_pnl_dollars": round(all_time, 4),
            "today_trade_count": today_count,
            "total_trade_count": len(_trades),
            "session_start_balance_dollars": _session_start_balance,
            "daily_history": {
                str(d): round(v, 4) for d, v in sorted(_daily_realized.items())
            },
        }


# ── Fills-based reconciliation ────────────────────────────────────────────────

async def reconcile_from_fills() -> dict:
    """
    Pull today's fills from Kalshi and compute realized P&L for any
    closed positions not yet recorded locally.

    Returns a summary of what was reconciled.
    """
    import client as kalshi

    try:
        resp = await kalshi.get_fills(limit=200)
        fills = resp.get("fills", [])
    except Exception as exc:
        logger.warning(f"Reconcile: failed to fetch fills: {exc}")
        return {"error": str(exc)}

    today_str = date.today().isoformat()

    # Group fills by ticker: { ticker: { "buys": [...], "sells": [...] } }
    by_ticker: dict[str, dict] = {}
    for f in fills:
        created = f.get("created_time", "")
        if not created.startswith(today_str):
            continue
        ticker = f.get("ticker") or f.get("market_ticker", "")
        if not ticker:
            continue
        grp = by_ticker.setdefault(ticker, {"buys": [], "sells": []})
        if f.get("action") == "buy":
            grp["buys"].append(f)
        elif f.get("action") == "sell":
            grp["sells"].append(f)

    # Find already-recorded tickers so we don't double-count
    with _lock:
        recorded_tickers = {t.ticker for t in _trades}

    reconciled = []
    for ticker, grp in by_ticker.items():
        buys = grp["buys"]
        sells = grp["sells"]
        if not buys:
            continue

        # Determine which side we hold
        # Kalshi fills: side="no" means we traded NO contracts
        # Use the first buy to establish side
        first_buy = buys[0]
        our_side = first_buy.get("side", "no")   # "yes" or "no"

        total_bought = sum(float(f.get("count_fp", 0)) for f in buys)
        total_sold   = sum(float(f.get("count_fp", 0)) for f in sells)

        # Only compute P&L for positions that have been fully exited
        if total_sold < total_bought * 0.95 and total_sold < 1:
            # Still open — check if market is settled
            try:
                mdata = await kalshi.get_market(ticker)
                mkt = mdata.get("market", mdata)
                if mkt.get("status") != "finalized":
                    continue   # genuinely still open
                result = mkt.get("result", "")
                if result not in ("yes", "no"):
                    continue
                settlement_yes = 100 if result == "yes" else 0
            except Exception:
                continue

            # Settled without an explicit sell — compute settlement P&L
            avg_buy_price = sum(
                float(f.get("no_price_dollars" if our_side == "no" else "yes_price_dollars", 0)) * 100
                * float(f.get("count_fp", 0))
                for f in buys
            ) / max(total_bought, 1)

            if our_side == "yes":
                pnl = (settlement_yes - avg_buy_price) / 100.0 * total_bought
            else:
                no_settle = 100 - settlement_yes
                pnl = (no_settle - avg_buy_price) / 100.0 * total_bought

            note = f"reconcile_settle_{result}"
        else:
            # Position was sold — compute round-trip P&L
            avg_buy_price = sum(
                float(f.get("no_price_dollars" if our_side == "no" else "yes_price_dollars", 0)) * 100
                * float(f.get("count_fp", 0))
                for f in buys
            ) / max(total_bought, 1)

            avg_sell_price = sum(
                float(f.get("no_price_dollars" if our_side == "no" else "yes_price_dollars", 0)) * 100
                * float(f.get("count_fp", 0))
                for f in sells
            ) / max(total_sold, 1)

            closed_count = min(total_bought, total_sold)
            if our_side == "yes":
                pnl = (avg_sell_price - avg_buy_price) / 100.0 * closed_count
            else:
                pnl = (avg_sell_price - avg_buy_price) / 100.0 * closed_count

            note = "reconcile_sell"

        # Skip if already recorded
        if ticker in recorded_tickers:
            logger.debug(f"Reconcile: {ticker} already recorded, skipping")
            continue

        filled_at = datetime.fromisoformat(
            first_buy.get("created_time", "").replace("Z", "+00:00")
        ).replace(tzinfo=None)

        record_trade(
            ticker, our_side, "reconcile",
            round(total_bought), avg_buy_price,
            pnl_dollars=round(pnl, 4), note=note,
        )
        reconciled.append({"ticker": ticker, "side": our_side, "pnl": round(pnl, 4), "note": note})
        logger.info(f"Reconciled {ticker}: side={our_side} P&L=${pnl:+.2f}")

    return {"reconciled": len(reconciled), "items": reconciled}
