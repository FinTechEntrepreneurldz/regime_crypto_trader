from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .execution import execute_orders, plan_orders
from .features import compute_regime_features
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
    "diagnostics",
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

def build_gate_diagnostics(close: pd.DataFrame, cfg: dict, signal, timestamp: str) -> tuple[pd.DataFrame, dict]:
    regime = compute_regime_features(close, cfg).sort_index()
    gate_cfg = cfg.get("gate", {})
    asof = pd.Timestamp(signal.asof_date)

    if regime.empty:
        empty = pd.DataFrame(columns=[
            "timestamp_utc", "market_date", "component", "value", "threshold",
            "operator", "required", "passed", "status", "description"
        ])
        return empty, {
            "computed_at": timestamp,
            "market_date": str(asof.date()),
            "active": bool(signal.active),
            "failed_conditions": ["regime_features_missing"],
            "summary": "Regime feature frame is empty.",
        }

    regime = regime.loc[:asof] if len(regime.loc[:asof]) else regime
    latest = regime.iloc[-1]
    market_date = str(pd.Timestamp(regime.index[-1]).date())

    risk_on_score_min = float(gate_cfg.get("risk_on_score_min", 0.65))
    breadth50_min = float(gate_cfg.get("breadth50_min", 0.55))
    min_btc_drawdown_63 = float(gate_cfg.get("min_btc_drawdown_63", -0.20))
    confirmation_days = int(gate_cfg.get("confirmation_days", 1))

    risk_on_score_pass = regime["risk_on_score"] >= risk_on_score_min
    breadth_pass = regime["breadth_50"] >= breadth50_min
    drawdown_pass = regime["btc_dd_63"] >= min_btc_drawdown_63
    btc50_pass = regime["btc_above_50"] > 0.5
    btc200_pass = regime["btc_above_200"] > 0.5
    eth50_pass = regime["eth_above_50"] > 0.5
    btc_ret_pass = regime["btc_ret_21"] > 0

    raw_gate = risk_on_score_pass & breadth_pass & drawdown_pass

    if gate_cfg.get("require_btc_above_50", True):
        raw_gate &= btc50_pass
    if gate_cfg.get("require_btc_above_200", True):
        raw_gate &= btc200_pass
    if gate_cfg.get("require_eth_above_50", True):
        raw_gate &= eth50_pass
    if gate_cfg.get("require_btc_ret_21_positive", True):
        raw_gate &= btc_ret_pass

    if confirmation_days > 1:
        confirmation_count = int(raw_gate.tail(confirmation_days).sum())
        confirmation_pass = confirmation_count >= confirmation_days
        final_gate = bool(raw_gate.rolling(confirmation_days).sum().iloc[-1] >= confirmation_days)
    else:
        confirmation_count = int(bool(raw_gate.iloc[-1]))
        confirmation_pass = bool(raw_gate.iloc[-1])
        final_gate = bool(raw_gate.iloc[-1])

    checks = [
        {
            "component": "risk_on_score",
            "value": float(latest.get("risk_on_score", 0.0)),
            "threshold": risk_on_score_min,
            "operator": ">=",
            "required": True,
            "passed": bool(risk_on_score_pass.iloc[-1]),
            "description": "Composite crypto risk-on score must be above the configured minimum.",
        },
        {
            "component": "breadth_50",
            "value": float(latest.get("breadth_50", 0.0)),
            "threshold": breadth50_min,
            "operator": ">=",
            "required": True,
            "passed": bool(breadth_pass.iloc[-1]),
            "description": "Universe breadth above the 50-day moving average must be strong enough.",
        },
        {
            "component": "btc_dd_63",
            "value": float(latest.get("btc_dd_63", 0.0)),
            "threshold": min_btc_drawdown_63,
            "operator": ">=",
            "required": True,
            "passed": bool(drawdown_pass.iloc[-1]),
            "description": "BTC 63-day drawdown must not be worse than the configured limit.",
        },
        {
            "component": "btc_above_50",
            "value": float(latest.get("btc_above_50", 0.0)),
            "threshold": 0.5,
            "operator": ">",
            "required": bool(gate_cfg.get("require_btc_above_50", True)),
            "passed": (not gate_cfg.get("require_btc_above_50", True)) or bool(btc50_pass.iloc[-1]),
            "description": "BTC must be above its 50-day moving average when required.",
        },
        {
            "component": "btc_above_200",
            "value": float(latest.get("btc_above_200", 0.0)),
            "threshold": 0.5,
            "operator": ">",
            "required": bool(gate_cfg.get("require_btc_above_200", True)),
            "passed": (not gate_cfg.get("require_btc_above_200", True)) or bool(btc200_pass.iloc[-1]),
            "description": "BTC must be above its 200-day moving average when required.",
        },
        {
            "component": "eth_above_50",
            "value": float(latest.get("eth_above_50", 0.0)),
            "threshold": 0.5,
            "operator": ">",
            "required": bool(gate_cfg.get("require_eth_above_50", True)),
            "passed": (not gate_cfg.get("require_eth_above_50", True)) or bool(eth50_pass.iloc[-1]),
            "description": "ETH must be above its 50-day moving average when required.",
        },
        {
            "component": "btc_ret_21_positive",
            "value": float(latest.get("btc_ret_21", 0.0)),
            "threshold": 0.0,
            "operator": ">",
            "required": bool(gate_cfg.get("require_btc_ret_21_positive", True)),
            "passed": (not gate_cfg.get("require_btc_ret_21_positive", True)) or bool(btc_ret_pass.iloc[-1]),
            "description": "BTC 21-day return must be positive when required.",
        },
        {
            "component": "confirmation_days",
            "value": confirmation_count,
            "threshold": confirmation_days,
            "operator": ">=",
            "required": True,
            "passed": bool(confirmation_pass),
            "description": f"Raw gate must pass for {confirmation_days} consecutive day(s).",
        },
        {
            "component": "final_v26_gate",
            "value": int(final_gate),
            "threshold": 1,
            "operator": "==",
            "required": True,
            "passed": bool(signal.active),
            "description": "Final V26 risk gate after all checks and confirmation-day logic.",
        },
    ]

    rows = []
    for check in checks:
        rows.append({
            "timestamp_utc": timestamp,
            "market_date": market_date,
            "component": check["component"],
            "value": check["value"],
            "threshold": check["threshold"],
            "operator": check["operator"],
            "required": check["required"],
            "passed": check["passed"],
            "status": "PASS" if check["passed"] else "FAIL",
            "description": check["description"],
        })

    failed = [
        row["component"]
        for row in rows
        if row["required"] and not row["passed"]
    ]

    summary = {
        "computed_at": timestamp,
        "market_date": market_date,
        "active": bool(signal.active),
        "raw_gate_today": bool(raw_gate.iloc[-1]),
        "confirmation_days_required": confirmation_days,
        "confirmation_days_passed_count": confirmation_count,
        "confirmation_days_pass": bool(confirmation_pass),
        "failed_conditions": failed,
        "regime_snapshot": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in latest.to_dict().items()
        },
        "summary": "V26 gate is ON." if signal.active else f"V26 gate is OFF. Failed checks: {', '.join(failed) if failed else 'unknown'}."
    }

    return pd.DataFrame(rows), summary

def health_status(
    signal,
    target_weights: pd.Series,
    planned: pd.DataFrame,
    submitted: pd.DataFrame,
    equity: float,
    timestamp: str,
    gate_summary: dict | None = None,
) -> dict:
    failed = 0

    if submitted is not None and not submitted.empty and "status" in submitted.columns:
        failed = int(
            submitted["status"]
            .astype(str)
            .str.lower()
            .str.contains("failed|error")
            .sum()
        )

    alerts = []

    if failed > 0:
        alerts.append(f"{failed} submitted order rows failed or errored.")

    if not signal.active:
        alerts.append("V26 risk-on gate is false; strategy target is flat.")

    if gate_summary and gate_summary.get("failed_conditions"):
        alerts.append(
            "Failed V26 gate checks: " + ", ".join(gate_summary["failed_conditions"])
        )

    out = {
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

    if gate_summary:
        out["gate_diagnostics"] = gate_summary

    return out

def write_canonical_logs(signal, target_weights: pd.Series, planned_orders: list[Any], submitted: pd.DataFrame, positions: pd.DataFrame, equity: float, cfg: dict, timestamp: str, close: pd.DataFrame | None = None) -> None:
    ensure_log_dirs()
    submit_orders = execute_orders_enabled(cfg)
    action = "risk_on_long_only" if signal.active else "risk_gate_off_flat"
    last_action = read_last_action()

    planned_df = order_rows_from_planned(planned_orders, timestamp)
    submitted_df = submitted_rows(submitted, timestamp)
    positions_df = normalize_positions(positions, timestamp)
    target_df = target_weight_rows(target_weights, timestamp)

    gate_df = pd.DataFrame()
    gate_summary = None
    if close is not None:
        gate_df, gate_summary = build_gate_diagnostics(close, cfg, signal, timestamp)

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
    if not gate_df.empty:
        write_latest_and_append(
            gate_df,
            LOG_DIR / "diagnostics" / "latest_gate_diagnostics.csv",
            LOG_DIR / "diagnostics" / "gate_diagnostics.csv",
        )
        (LOG_DIR / "diagnostics" / "latest_gate_diagnostics.json").write_text(
            json.dumps(gate_summary, indent=2, sort_keys=True)
        )
    append_csv(portfolio_row(equity, action, submit_orders, timestamp), LOG_DIR / "portfolio" / "portfolio.csv")
    append_csv(signal_history_row(signal, target_weights, equity, timestamp), LOG_DIR / "health" / "signal_history.csv")
    (LOG_DIR / "health" / "health_status.json").write_text(
        json.dumps(
            health_status(signal, target_weights, planned_df, submitted_df, equity, timestamp, gate_summary),
            indent=2,
            sort_keys=True,
        )
    )


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
    write_canonical_logs(signal, target_weights, planned_orders, order_df, post_positions, equity, cfg, timestamp, close=close)

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
