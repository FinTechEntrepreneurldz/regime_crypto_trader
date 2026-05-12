from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .features import asset_score_frame, compute_regime_features, compute_returns, v26_gate


@dataclass
class SignalResult:
    asof_date: pd.Timestamp
    active: bool
    target_weights: pd.Series
    regime_snapshot: Dict
    selected_assets: List[str]
    reason: str


def _inverse_vol_weights(returns: pd.DataFrame, assets: List[str], asof, cfg: dict) -> pd.Series:
    if not assets:
        return pd.Series(dtype=float)
    vol_window = int(cfg["portfolio"].get("vol_lookback", 63))
    hist = returns.loc[:asof, assets].tail(vol_window)
    vol = hist.std().replace(0, np.nan)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() <= 0:
        inv = pd.Series(1.0, index=assets)
    w = inv / inv.sum()
    return w


def _cap_weights(w: pd.Series, cap: float, gross: float) -> pd.Series:
    w = w.clip(lower=0).copy()
    if w.sum() <= 0:
        return w
    w = w / w.sum() * gross
    for _ in range(25):
        over = w > cap
        if not over.any(): break
        fixed = w[over].clip(upper=cap)
        free = w[~over]
        remaining = gross - fixed.sum()
        if remaining <= 0 or free.empty:
            return fixed.reindex(w.index).fillna(0.0)
        free = free / free.sum() * remaining if free.sum() > 0 else pd.Series(remaining / len(free), index=free.index)
        w = pd.concat([fixed, free]).reindex(w.index).fillna(0.0)
    return w


def _portfolio_vol_scale(returns: pd.DataFrame, weights: pd.Series, asof, cfg: dict) -> float:
    if weights.empty or weights.sum() <= 0:
        return 0.0
    target_vol = float(cfg["portfolio"].get("target_vol", 0.08))
    max_gross = float(cfg["portfolio"].get("max_gross_exposure", 0.75))
    window = int(cfg["portfolio"].get("vol_lookback", 63))
    hist = returns.loc[:asof, weights.index].tail(window)
    if len(hist) < 20:
        return min(0.25, max_gross)
    port = hist.dot(weights)
    vol = float(port.std() * np.sqrt(252))
    if vol <= 1e-12:
        return min(0.25, max_gross)
    return float(np.clip(target_vol / vol, 0.0, max_gross))


def generate_target_weights(close: pd.DataFrame, cfg: dict, asof_date=None) -> SignalResult:
    close = close.sort_index().ffill()
    if asof_date is None:
        asof = pd.Timestamp(close.index[-1])
    else:
        asof = pd.Timestamp(asof_date)
        close = close.loc[:asof]
    returns = compute_returns(close)
    regime = compute_regime_features(close, cfg)
    gate = v26_gate(regime, cfg)
    active = bool(gate.iloc[-1])
    snapshot = regime.iloc[-1].to_dict()
    if not active:
        return SignalResult(asof, False, pd.Series(0.0, index=close.columns), snapshot, [], "V26 risk-on gate is false; target is flat/zero new exposure.")

    scores = asset_score_frame(close, cfg).loc[asof].dropna().sort_values(ascending=False)
    long_n = int(cfg["portfolio"].get("long_n", 6))
    selected = scores.head(long_n).index.tolist()
    base_w = _inverse_vol_weights(returns, selected, asof, cfg)
    gross_scale = _portfolio_vol_scale(returns, base_w, asof, cfg)
    max_single = float(cfg["portfolio"].get("max_single_asset_weight", 0.20))
    w = _cap_weights(base_w, max_single, gross_scale)
    min_w = float(cfg["portfolio"].get("min_target_weight", 0.015))
    w = w.where(w >= min_w, 0.0)
    if w.sum() > 0:
        w = _cap_weights(w, max_single, min(float(cfg["portfolio"].get("max_gross_exposure", 0.75)), w.sum()))
    out = pd.Series(0.0, index=close.columns)
    out.loc[w.index] = w.values
    return SignalResult(asof, True, out.sort_values(ascending=False), snapshot, selected, "V26 risk-on gate true; long-only spot weights generated.")