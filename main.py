"""
Kalshi Trading API — FastAPI service.

Endpoints:
  GET  /health                      — liveness check
  GET  /balance                     — portfolio cash balance
  GET  /markets                     — fetch open markets (paginated)
  GET  /markets/scan                — scan for mispriced contracts
  GET  /markets/{ticker}            — market detail + orderbook
  POST /orders                      — place a buy/sell order (optionally with SL/PT)
  GET  /orders                      — list recent orders
  DELETE /orders/{order_id}         — cancel an order
  GET  /positions                   — all Kalshi positions + managed state
  GET  /positions/managed           — only positions under SL/PT management
  PUT  /positions/{ticker}          — update SL/PT thresholds
  DELETE /positions/{ticker}        — remove from management
  POST /monitor                     — manually trigger one monitor cycle
  GET  /autopilot                   — autopilot status + config
  POST /autopilot/start             — enable autopilot
  POST /autopilot/stop              — disable autopilot
  PUT  /autopilot/config            — update autopilot parameters
  POST /autopilot/run               — manually trigger one autopilot cycle
  GET  /pnl                         — P&L summary and trade history
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import autopilot
import client as kalshi
import pnl as pnl_tracker
import positions as pos_mgr
import scanner
from models import (
    AutopilotConfigRequest,
    CancelOrderRequest,
    OrderRequest,
    PositionUpdateRequest,
    ScanRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("main")

MONITOR_INTERVAL_SEC = int(os.getenv("MONITOR_INTERVAL_SEC", "30"))
_monitor_task: asyncio.Task | None = None


async def _monitor_loop():
    while True:
        try:
            actions = await pos_mgr.monitor_once()
            if actions:
                logger.info(f"Monitor actions: {actions}")
        except Exception as exc:
            logger.warning(f"Monitor cycle error: {exc}")
        await asyncio.sleep(MONITOR_INTERVAL_SEC)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor_task

    # Restore managed positions from disk (survives restarts)
    pos_mgr.load_from_disk()

    # Restore autopilot config (crypto_enabled, budgets, etc.)
    autopilot.load_config()

    # Restore P&L history from disk
    pnl_tracker.load_from_disk()

    # Record starting balance for P&L baseline (only sets if not already saved)
    try:
        bal = await kalshi.get_balance()
        pnl_tracker.set_session_start_balance(float(bal.get("balance_dollars", 0)))
    except Exception as exc:
        logger.warning(f"Could not fetch starting balance: {exc}")

    # Reconcile any positions settled/closed since last run
    try:
        rec = await pnl_tracker.reconcile_from_fills()
        if rec.get("reconciled", 0):
            logger.info(f"Startup reconciliation: {rec}")
    except Exception as exc:
        logger.warning(f"Startup reconciliation failed: {exc}")

    _monitor_task = asyncio.create_task(_monitor_loop())
    autopilot.config.enabled = True   # always start hot — toggle via API/dashboard
    autopilot.start_task()
    logger.info(f"Services started — autopilot ON (monitor interval={MONITOR_INTERVAL_SEC}s)")
    yield
    if _monitor_task:
        _monitor_task.cancel()


# ── App ───────────────────────────────────────────────────────────────────────

BASE_PATH = os.getenv("BASE_PATH", "/kalshi").rstrip("/")

app = FastAPI(
    title="Kalshi Trader",
    description="Scan Kalshi markets, find mispriced contracts, manage orders and positions.",
    version="1.0.0",
    lifespan=lifespan,
    root_path=BASE_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware


class StripPrefixMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope.get("path", "")
        if BASE_PATH and path.startswith(BASE_PATH):
            stripped = path[len(BASE_PATH):] or "/"
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)


if BASE_PATH:
    app.add_middleware(StripPrefixMiddleware)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(STATIC_DIR / "index.html")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


# ── Markets ───────────────────────────────────────────────────────────────────

@app.get("/markets", tags=["markets"])
async def list_markets(
    limit: int = Query(100, ge=1, le=1000),
    cursor: str | None = Query(None),
    status: str = Query("open"),
):
    try:
        return await kalshi.get_markets(status=status, limit=limit, cursor=cursor)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/markets/scan", tags=["markets"])
async def scan_markets(
    max_events: int = Query(100, ge=1, le=500),
    min_score: float = Query(1.0, ge=0),
    category: str | None = Query(None, description="e.g. 'Politics', 'Economics', 'Financials'"),
    event_ticker: str | None = Query(None, description="Scan only a specific event"),
):
    """
    Scan open Kalshi markets for mispricing signals, sorted by score.
    """
    try:
        results, scanned = await scanner.scan_markets(
            max_events=max_events,
            min_score=min_score,
            category=category,
            event_ticker=event_ticker,
        )
        return {"count": len(results), "scanned": scanned, "markets": results}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/markets/{ticker}", tags=["markets"])
async def market_detail(ticker: str):
    try:
        return await scanner.get_market_detail(ticker)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Orders ────────────────────────────────────────────────────────────────────

@app.post("/orders", tags=["orders"])
async def place_order(req: OrderRequest):
    try:
        resp = await kalshi.place_order(
            ticker=req.ticker,
            side=req.side,
            action=req.action,
            order_type=req.order_type,
            count=req.count,
            yes_price=req.yes_price,
            client_order_id=req.client_order_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    managed = None
    if (req.stop_loss is not None or req.profit_target is not None) and req.action == "buy":
        entry = req.yes_price or 50
        sl = req.stop_loss or max(1, entry - 20)
        pt = req.profit_target or min(99, entry + 20)
        managed = pos_mgr.add_position(
            ticker=req.ticker, side=req.side, contracts=req.count,
            entry_price=entry, stop_loss=sl, profit_target=pt,
        )

    return {
        "order": resp,
        "managed": pos_mgr._position_to_dict(managed) if managed else None,
    }


@app.get("/orders", tags=["orders"])
async def list_orders(status: str | None = Query(None)):
    try:
        return await kalshi.get_orders(status=status)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.delete("/orders/{order_id}", tags=["orders"])
async def cancel_order(order_id: str):
    try:
        return await kalshi.cancel_order(order_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Positions ─────────────────────────────────────────────────────────────────

@app.get("/positions", tags=["positions"])
async def list_positions():
    try:
        kalshi_data = await kalshi.get_positions()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    managed_map = {p["ticker"]: p for p in pos_mgr.get_all_positions()}
    positions = kalshi_data.get("market_positions", [])
    enriched = [{**p, "managed": managed_map.get(p.get("ticker", ""))} for p in positions]
    return {"positions": enriched, "managed_count": len(managed_map)}


@app.get("/positions/managed", tags=["positions"])
async def list_managed_positions():
    return {"positions": pos_mgr.get_all_positions()}


@app.put("/positions/{ticker}", tags=["positions"])
async def update_position(ticker: str, req: PositionUpdateRequest):
    position = pos_mgr.get_position(ticker)
    if not position:
        raise HTTPException(status_code=404, detail=f"No managed position for {ticker}")
    if req.stop_loss is not None:
        position.stop_loss = req.stop_loss
    if req.profit_target is not None:
        position.profit_target = req.profit_target
    return pos_mgr._position_to_dict(position)


@app.delete("/positions/{ticker}", tags=["positions"])
async def remove_managed_position(ticker: str):
    removed = pos_mgr.remove_position(ticker)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No managed position for {ticker}")
    return {"removed": ticker}


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/balance", tags=["portfolio"])
async def get_balance():
    try:
        return await kalshi.get_balance()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── P&L ───────────────────────────────────────────────────────────────────────

@app.get("/pnl", tags=["portfolio"])
async def get_pnl():
    """P&L summary, daily history, and trade log for today."""
    summary = pnl_tracker.summary()
    trades = [
        {
            "ticker": t.ticker,
            "side": t.side,
            "action": t.action,
            "contracts": t.contracts,
            "price_cents": t.price_cents,
            "pnl_dollars": t.pnl_dollars,
            "note": t.note,
            "filled_at": t.filled_at.isoformat(),
        }
        for t in pnl_tracker.today_trades()
    ]
    unrealized_total, unrealized_by_ticker = pos_mgr.get_unrealized_pnl()
    realized = summary["today_realized_pnl_dollars"]
    return {
        **summary,
        "today_trades": trades,
        "unrealized_pnl_dollars": unrealized_total,
        "unrealized_by_ticker": unrealized_by_ticker,
        "total_pnl_dollars": round(realized + unrealized_total, 4),
    }


@app.post("/pnl/reconcile", tags=["portfolio"])
async def reconcile_pnl():
    """Pull today's Kalshi fills and backfill any realized P&L not yet recorded locally."""
    try:
        result = await pnl_tracker.reconcile_from_fills()
        return result
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Monitor ───────────────────────────────────────────────────────────────────

@app.post("/monitor", tags=["system"])
async def trigger_monitor():
    try:
        actions = await pos_mgr.monitor_once()
        return {"actions": actions, "checked": len(pos_mgr.get_all_positions())}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


# ── Autopilot ─────────────────────────────────────────────────────────────────

@app.get("/autopilot", tags=["autopilot"])
async def autopilot_status():
    """Current autopilot configuration and P&L status."""
    return autopilot.get_status()


@app.post("/autopilot/start", tags=["autopilot"])
async def autopilot_start():
    """Enable the autopilot. It will scan and trade on the configured interval."""
    autopilot.config.enabled = True
    logger.info("Autopilot ENABLED")
    return {"enabled": True, "config": autopilot.get_status()}


@app.post("/autopilot/stop", tags=["autopilot"])
async def autopilot_stop():
    """Disable the autopilot. Open positions are NOT closed."""
    autopilot.config.enabled = False
    logger.info("Autopilot DISABLED")
    return {"enabled": False}


@app.put("/autopilot/config", tags=["autopilot"])
async def autopilot_configure(req: AutopilotConfigRequest):
    """
    Update any autopilot parameter without restarting.
    Only fields you include are changed — omitted fields stay the same.
    """
    cfg = autopilot.config
    if req.enabled is not None:
        cfg.enabled = req.enabled
    if req.scan_interval_sec is not None:
        cfg.scan_interval_sec = req.scan_interval_sec
    if req.daily_loss_limit_usd is not None:
        cfg.daily_loss_limit_usd = req.daily_loss_limit_usd
    if req.daily_budget_usd is not None:
        cfg.daily_budget_usd = req.daily_budget_usd
    if req.max_trade_usd is not None:
        cfg.max_trade_usd = req.max_trade_usd
    if req.contracts_per_trade is not None:
        cfg.contracts_per_trade = req.contracts_per_trade
    if req.min_score is not None:
        cfg.min_score = req.min_score
    if req.arb_threshold_cents is not None:
        cfg.arb_threshold_cents = req.arb_threshold_cents
    if req.drift_min_cents is not None:
        cfg.drift_min_cents = req.drift_min_cents
    if req.stop_loss_pct is not None:
        cfg.stop_loss_pct = req.stop_loss_pct
    if req.profit_target_pct is not None:
        cfg.profit_target_pct = req.profit_target_pct
    if req.max_events_to_scan is not None:
        cfg.max_events_to_scan = req.max_events_to_scan
    if req.scan_category is not None:
        cfg.scan_category = req.scan_category if req.scan_category != "" else None
    if req.crypto_enabled is not None:
        cfg.crypto_enabled = req.crypto_enabled
        logger.info(f"Crypto mode {'ENABLED' if cfg.crypto_enabled else 'DISABLED'}")
    autopilot.save_config()
    return autopilot.get_status()


@app.get("/crypto/signal", tags=["crypto"])
async def get_crypto_signal():
    """Fetch live BTC/ETH candles + crypto news sentiment from Binance/CryptoCompare."""
    import crypto_signals as csig
    try:
        sig = await csig.get_crypto_signal()
        return sig.to_dict()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/autopilot/run", tags=["autopilot"])
async def autopilot_run_now():
    """Trigger a scan cycle in the background; returns last completed result immediately."""
    triggered = autopilot.trigger_cycle_background()
    return {
        "triggered": triggered,
        "message": "Cycle already running" if not triggered else "Cycle started in background",
        "last_result": autopilot._last_result,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
