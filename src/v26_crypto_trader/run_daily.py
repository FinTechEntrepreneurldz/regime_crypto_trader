from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .alpaca_data import bars_to_close_volume, load_crypto_bars_alpaca
from .broker_alpaca import get_account_info, get_positions, trading_client
from .execution import execute_orders, plan_orders
from .performance import snapshot_performance
from .risk import apply_risk_limits
from .settings import execute_orders_enabled, load_config, model_notional
from .strategy import generate_target_weights

MODEL_ID = "regime_crypto_trader"
MODEL_NAME = "Regime Crypto Trader — V26 Long-Only"
ARTIFACT_PATH = "model_artifacts/selected_strategy.json"
METADATA_PATH = "model_artifacts/metadata.json"
LOG_DIR = Path("logs")

CANONICAL_DIRS = [
    "decisions",
    "orders",
    "positions",
    "portfolio",
    "target_weights",
    "health",
]


def utc_now() -> str:
    return pd.Timestamp.utcnow().isoformat()


def ensure_log_dirs() -> None:
    for name in CANONICAL_DIRS:
        (LOG_DIR / name).mkdir(parents=True, exist_ok=True)


def append_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=header, index=False)


def write_latest_and_append(df: pd.DataFrame, latest_path: str | Path, history_path: str | Path) -> None:
    latest_path = Path(latest_path)
    history_path = Path(history_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(latest_path, index=False)
    append_csv(df, history_path)


def read_last_action() -> str:
    latest_path = LOG_DIR / "decisions" / "latest_decision.csv"
    if not latest_path.exists():
        return ""
    try:
        df = pd.read_csv(latest_path)
        if df.empty:
            return ""
        return str(df.iloc[-1].get("action", ""))
    except Exception:
        return ""


def normalize_positions(positions: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    if positions is None or positions.empty:
        return pd.DataFrame(columns=["symbol_raw", "symbol", "qty", "market_value", "current_price", "unrealized_pl", "asset_class", "side", "timestamp_utc"])
    out = positions.copy()
    if "side" not in out.columns:
        out["side"] = out["qty"].astype(float).map(lambda q: "long" if q >= 0 else "short")
    out["timestamp_utc"] = timestamp
    return out


def order_rows_from_planned(orders: list[Any], timestamp: str) -> pd.DataFrame:
    rows = []
    for o in orders:
        rows.append({
            "symbol": o.symbol,
            "side": o.side,
            "qty": o.qty,
            "notional": o.notional,
            "current_notional": o.current_notional,
            "target_notional": o.target_notional,
            "delta_notional": o.delta_notional,
            "reason": o.reason,
            "timestamp_utc": timestamp,
        })
    return pd.DataFrame(rows, columns=["symbol", "side", "qty", "notional", "current_notional", "target_notional", "delta_notional", "reason", "timestamp_utc"])


def submitted_rows(order_df: pd.DataFrame, timestamp: str) -> pd.DataFrame:
    if order_df is None or order_df.empty:
        return pd.DataFrame(columns=["symbol", "side", "qty", "notional", "current_notional", "target_notional", "delta_notional", "reason", "execute", "status", "order_id", "error", "timestamp_utc"])
    out = order_df.copy()
    out["timestamp_utc"] = timestamp
    for col in ["error", "order_id"]:
        if col not in out.columns:
            out[col] = ""
    return out


def target_weight_rows(target_weights: pd.Series, timestamp: str) -> pd.DataFrame:
    active = target_weights[target_weights.abs() > 0].sort_values(ascending=False)
    rows = pd.DataFrame({"symbol": active.index, "target_weight": active.values})
    rows["timestamp_utc"] = timestamp
    return rows


def portfolio_row(equity: float, action: str, submit_orders: bool, timestamp: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "timestamp_utc": timestamp,
        "portfolio_value": float(equity),
        "action": action,
        "submit_orders": bool(submit_orders),
    }])


def signal_history_row(signal, target_weights: pd.Series, equity: float, timestamp: str) -> pd.DataFrame:
    row = {
        "timestamp_utc": timestamp,
        "date": str(signal.asof_date.date()),
        "status": "ok",
        "active": bool(signal.active),
        "reason": signal.reason,
        "selected_assets": ",".join(signal.selected_assets),
        "account_value": float(equity),
        "n_target_positions": int((target_weights.abs() > 0).sum()),
        "gross_target_weight": float(target_weights.abs().sum()),
    }
    for key, value in signal.regime_snapshot.items():
        row[f"regime_{key}"] = value
    return pd.DataFrame([row])


def health_status(signal, target_weights: pd.Series, planned: pd.DataFrame, submitted: pd.DataFrame, equity: float, timestamp: str) -> dict:
    failed = 0
    if submitted is not None and not submitted.empty and "status" in submitted.columns:
        failed = int(submitted["status"].astype(str).str.lower().str.contains("failed|error").sum())
    alerts = []
    if failed > 0:
        alerts.append(f"{failed} submitted order rows failed or errored.")
    if not signal.active:
        alerts.append("V26 risk-on gate is false; strategy target is flat.")
    return {
        "computed_at": timestamp,
        "overall_status": "degraded" if failed > 0 else "ok",
        "alerts": alerts,
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "active_signal": bool(signal.active),
        "risk_on_score": signal.regime_snapshot.get("risk_on_score"),
        "n_target_positions": int((target_weights.abs() > 0).sum()),
        "gross_target_weight": float(target_weights.abs().sum()),
        "account_value": float(equity),
        "n_orders_planned": int(0 if planned is None else len(planned)),
        "n_orders_submitted": int(0 if submitted is None else len(submitted)),
        "training_recommended": False,
    }


def write_canonical_logs(signal, target_weights: pd.Series, planned_orders: list[Any], submitted: pd.DataFrame, positions: pd.DataFrame, equity: float, cfg: dict, timestamp: str) -> None:
    ensure_log_dirs()
    submit_orders = execute_orders_enabled(cfg)
    action = "risk_on_long_only" if signal.active else "risk_gate_off_flat"
    last_action = read_last_action()

    planned_df = order_rows_from_planned(planned_orders, timestamp)
    submitted_df = submitted_rows(submitted, timestamp)
    positions_df = normalize_positions(positions, timestamp)
    target_df = target_weight_rows(target_weights, timestamp)

    decision = pd.DataFrame([{
        "market_date": str(signal.asof_date.date()),
        "variant": MODEL_NAME,
        "action": action,
        "action_idx": 1 if signal.active else 0,
        "last_action": last_action,
        "submit_orders": submit_orders,
        "account_status": "connected",
        "account_value": float(equity),
        "n_target_positions": int((target_weights.abs() > 0).sum()),
        "n_orders_planned": int(len(planned_df)),
        "n_orders_submitted": int(len(submitted_df)),
        "model_path": ARTIFACT_PATH,
        "metadata_path": METADATA_PATH,
        "timestamp_utc": timestamp,
    }])

    write_latest_and_append(decision, LOG_DIR / "decisions" / "latest_decision.csv", LOG_DIR / "decisions" / "decisions.csv")
    write_latest_and_append(target_df, LOG_DIR / "target_weights" / "latest_target_weights.csv", LOG_DIR / "target_weights" / "target_weights.csv")
    write_latest_and_append(planned_df, LOG_DIR / "orders" / "latest_planned_orders.csv", LOG_DIR / "orders" / "planned_orders.csv")
    write_latest_and_append(submitted_df, LOG_DIR / "orders" / "latest_submitted_orders.csv", LOG_DIR / "orders" / "submitted_orders.csv")
    write_latest_and_append(positions_df, LOG_DIR / "positions" / "latest_positions.csv", LOG_DIR / "positions" / "positions.csv")
    append_csv(portfolio_row(equity, action, submit_orders, timestamp), LOG_DIR / "portfolio" / "portfolio.csv")
    append_csv(signal_history_row(signal, target_weights, equity, timestamp), LOG_DIR / "health" / "signal_history.csv")
    (LOG_DIR / "health" / "health_status.json").write_text(json.dumps(health_status(signal, target_weights, planned_df, submitted_df, equity, timestamp), indent=2, sort_keys=True))


def main(config_path="config/frozen_config.yaml"):
    cfg = load_config(config_path)
    ensure_log_dirs()

    symbols = cfg["universe"]["symbols"]
    bars = load_crypto_bars_alpaca(symbols, lookback_days=int(cfg["universe"].get("lookback_days", 430)))
    close, volume = bars_to_close_volume(bars)

    signal = generate_target_weights(close, cfg)
    target_weights = apply_risk_limits(signal.target_weights, cfg)

    client = trading_client()
    planned_orders, pre_positions, pre_equity = plan_orders(target_weights, cfg, client=client)
    order_df = execute_orders(planned_orders, cfg, client=client)

    # Crypto orders often fill quickly, but the account endpoint can lag.
    if execute_orders_enabled(cfg) and not order_df.empty:
        time.sleep(8)

    try:
        account_info = get_account_info(client)
        post_positions = get_positions(client)
        equity = float(account_info.get("equity", pre_equity))
    except Exception:
        post_positions = pre_positions
        equity = float(pre_equity or model_notional(cfg))

    timestamp = utc_now()
    write_canonical_logs(signal, target_weights, planned_orders, order_df, post_positions, equity, cfg, timestamp)

    # Keep original lightweight performance outputs too.
    perf_df = snapshot_performance(equity, post_positions, {"active": signal.active, "risk_on_score": signal.regime_snapshot.get("risk_on_score")})
    legacy_dir = LOG_DIR / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    perf_df.to_csv(legacy_dir / "performance_latest.csv", index=False)

    summary = {
        "model_id": MODEL_ID,
        "asof_date": str(signal.asof_date.date()),
        "active": signal.active,
        "reason": signal.reason,
        "risk_on_score": signal.regime_snapshot.get("risk_on_score"),
        "selected_assets": signal.selected_assets,
        "target_weights": {k: float(v) for k, v in target_weights[target_weights > 0].items()},
        "account_value": equity,
        "orders_planned": len(planned_orders),
        "orders_submitted_or_logged": len(order_df),
        "execute_orders": execute_orders_enabled(cfg),
    }

    print("Regime Crypto Trader V26 run complete")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
