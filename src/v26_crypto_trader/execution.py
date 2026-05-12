from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from .broker_alpaca import get_account_equity, get_positions, submit_market_order, trading_client
from .risk import weights_to_target_notional
from .settings import execute_orders_enabled, model_notional


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    notional: float = 0.0
    qty: float | None = None
    reason: str = ""
    current_notional: float = 0.0
    target_notional: float = 0.0
    delta_notional: float = 0.0


def plan_orders(target_weights: pd.Series, cfg: dict, client=None) -> tuple[list[PlannedOrder], pd.DataFrame, float]:
    client = client or trading_client()
    account_equity = get_account_equity(client)
    sizing_equity = model_notional(cfg) or account_equity
    current = get_positions(client)

    target_notional = weights_to_target_notional(target_weights, sizing_equity)

    current_mv = pd.Series(dtype=float)
    current_qty = pd.Series(dtype=float)
    if not current.empty:
        current_mv = current.set_index("symbol")["market_value"].astype(float)
        current_qty = current.set_index("symbol")["qty"].astype(float)

    all_symbols = sorted(set(target_notional.index) | set(current_mv.index))
    orders: list[PlannedOrder] = []

    min_notional = float(cfg["risk"].get("min_order_notional", 25))
    max_order = float(cfg["risk"].get("max_order_notional", 250000))
    threshold_weight = float(cfg["portfolio"].get("rebalance_threshold_weight", 0.005))
    threshold_notional = max(min_notional, threshold_weight * sizing_equity)

    for sym in all_symbols:
        tgt = float(target_notional.get(sym, 0.0))
        cur = float(current_mv.get(sym, 0.0))
        delta = tgt - cur

        if abs(delta) < threshold_notional:
            continue

        if delta > 0:
            orders.append(
                PlannedOrder(
                    sym,
                    "buy",
                    notional=min(delta, max_order),
                    reason="raise long-only crypto exposure to target",
                    current_notional=cur,
                    target_notional=tgt,
                    delta_notional=delta,
                )
            )
        else:
            qty_avail = float(current_qty.get(sym, 0.0))
            if qty_avail <= 0:
                # Never short crypto on Alpaca spot.
                continue
            price = abs(cur / qty_avail) if qty_avail else 0.0
            qty_to_sell = min(qty_avail, abs(delta) / price) if price > 0 else qty_avail
            if qty_to_sell > 0:
                orders.append(
                    PlannedOrder(
                        sym,
                        "sell",
                        qty=qty_to_sell,
                        reason="reduce or flatten existing long-only crypto exposure",
                        current_notional=cur,
                        target_notional=tgt,
                        delta_notional=delta,
                    )
                )

    return orders, current, account_equity


def execute_orders(orders: List[PlannedOrder], cfg: dict, client=None) -> pd.DataFrame:
    execute = execute_orders_enabled(cfg)
    client = client or trading_client()
    rows = []

    for o in orders:
        row = {
            "symbol": o.symbol,
            "side": o.side,
            "qty": o.qty,
            "notional": o.notional,
            "current_notional": o.current_notional,
            "target_notional": o.target_notional,
            "delta_notional": o.delta_notional,
            "reason": o.reason,
            "execute": execute,
            "status": "planned",
            "order_id": "",
            "error": "",
        }
        if execute:
            try:
                resp = submit_market_order(
                    client,
                    o.symbol,
                    o.side,
                    notional=o.notional if o.side == "buy" else None,
                    qty=o.qty if o.side == "sell" else None,
                )
                row.update({"status": str(getattr(resp, "status", "submitted")), "order_id": str(resp.id)})
            except Exception as exc:
                row.update({"status": "failed", "error": repr(exc)})
        rows.append(row)

    return pd.DataFrame(rows)


def append_log(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)
