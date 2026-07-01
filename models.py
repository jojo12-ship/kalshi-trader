"""
Pydantic request/response models.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    ticker: str = Field(..., description="Kalshi market ticker, e.g. 'KXELONMARS-99'")
    side: str = Field(..., description="'yes' or 'no'")
    action: str = Field("buy", description="'buy' or 'sell'")
    order_type: str = Field("limit", description="'limit' or 'market'")
    count: int = Field(..., ge=1, description="Number of contracts")
    yes_price: int | None = Field(
        None, ge=1, le=99,
        description="Limit price in cents (required for limit orders)",
    )
    stop_loss: int | None = Field(
        None, ge=1, le=99,
        description="yes-price in cents to trigger stop-loss exit",
    )
    profit_target: int | None = Field(
        None, ge=1, le=99,
        description="yes-price in cents to trigger profit-target exit",
    )
    client_order_id: str | None = None


class CancelOrderRequest(BaseModel):
    order_id: str


class ScanRequest(BaseModel):
    limit: int = Field(200, ge=1, le=1000, description="Max markets to fetch")
    min_score: float = Field(5.0, ge=0, description="Minimum mispricing score to return")


class PositionUpdateRequest(BaseModel):
    stop_loss: int | None = Field(None, ge=1, le=99)
    profit_target: int | None = Field(None, ge=1, le=99)


class AutopilotConfigRequest(BaseModel):
    enabled: bool | None = None
    scan_interval_sec: int | None = Field(None, ge=30, le=3600)
    daily_loss_limit_usd: float | None = Field(None, ge=1.0, le=500.0)
    daily_budget_usd: float | None = Field(None, ge=1.0, le=1000.0)
    max_trade_usd: float | None = Field(None, ge=0.50, le=100.0)
    contracts_per_trade: int | None = Field(None, ge=1, le=100)
    min_score: float | None = Field(None, ge=0.0)
    arb_threshold_cents: float | None = Field(None, ge=80.0, le=99.9)
    drift_min_cents: float | None = Field(None, ge=5.0, le=50.0)
    stop_loss_pct: float | None = Field(None, ge=0.05, le=0.95)
    profit_target_pct: float | None = Field(None, ge=0.05, le=5.0)
    max_events_to_scan: int | None = Field(None, ge=10, le=500)
    scan_category: str | None = None
    crypto_enabled: bool | None = None
