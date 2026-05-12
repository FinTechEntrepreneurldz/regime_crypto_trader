from __future__ import annotations

import os
import pandas as pd

from .settings import get_env_bool


def apply_risk_limits(target_weights: pd.Series, cfg: dict, account_equity: float | None = None) -> pd.Series:
    w = target_weights.copy().fillna(0.0)
    # Alpaca crypto spot: long-only. Never submit negative target weights.
    w = w.clip(lower=0.0)
    if get_env_bool(cfg["risk"].get("hard_kill_switch_env", "STRATEGY_KILL_SWITCH"), False):
        return w * 0.0
    max_single = float(cfg["risk"].get("max_single_position_weight", cfg["portfolio"].get("max_single_asset_weight", 0.20)))
    max_gross = float(cfg["risk"].get("max_account_gross_exposure", cfg["portfolio"].get("max_gross_exposure", 0.75)))
    w = w.clip(upper=max_single)
    if w.sum() > max_gross and w.sum() > 0:
        w = w / w.sum() * max_gross
    cash_buffer = float(cfg["portfolio"].get("cash_buffer", 0.05))
    if w.sum() > (1.0 - cash_buffer):
        w = w / w.sum() * (1.0 - cash_buffer)
    return w.sort_values(ascending=False)


def weights_to_target_notional(weights: pd.Series, equity: float) -> pd.Series:
    return weights.fillna(0.0) * float(equity)