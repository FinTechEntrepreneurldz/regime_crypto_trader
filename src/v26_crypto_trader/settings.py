from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv


def load_config(config_path: str | Path = "config/frozen_config.yaml") -> Dict[str, Any]:
    load_dotenv()
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def first_env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def execute_orders_enabled(cfg: dict) -> bool:
    primary = cfg.get("alpaca", {}).get("execute_paper_orders_env", "EXECUTE_PAPER_ORDERS")
    return get_env_bool(primary, False) or get_env_bool("SUBMIT_ORDERS", False)


def model_notional(cfg: dict) -> float:
    configured = float(cfg.get("portfolio", {}).get("model_notional", 1_000_000))
    return get_env_float("MODEL_NOTIONAL", get_env_float("DEFAULT_ACCOUNT_VALUE", configured))
