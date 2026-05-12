from __future__ import annotations

import numpy as np
import pandas as pd


def clean_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    try:
        out.index = out.index.tz_localize(None)
    except Exception:
        pass
    return out.sort_index()


def compute_returns(close: pd.DataFrame) -> pd.DataFrame:
    return clean_index(close).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def max_drawdown_from_price(price: pd.Series, window: int = 63) -> pd.Series:
    roll_max = price.rolling(window, min_periods=max(10, window // 3)).max()
    return price / roll_max - 1.0


def compute_regime_features(close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close = clean_index(close).ffill()
    symbols = list(close.columns)
    btc = close["BTC/USD"] if "BTC/USD" in symbols else close.iloc[:, 0]
    eth = close["ETH/USD"] if "ETH/USD" in symbols else btc
    ma_fast = int(cfg["features"].get("ma_fast", 50))
    ma_slow = int(cfg["features"].get("ma_slow", 200))
    breadth50 = (close > close.rolling(ma_fast, min_periods=max(10, ma_fast // 2)).mean()).mean(axis=1)
    out = pd.DataFrame(index=close.index)
    out["btc_above_50"] = (btc > btc.rolling(ma_fast, min_periods=max(10, ma_fast // 2)).mean()).astype(float)
    out["btc_above_200"] = (btc > btc.rolling(ma_slow, min_periods=max(50, ma_slow // 2)).mean()).astype(float)
    out["eth_above_50"] = (eth > eth.rolling(ma_fast, min_periods=max(10, ma_fast // 2)).mean()).astype(float)
    out["breadth_50"] = breadth50.fillna(0.0)
    out["btc_ret_21"] = btc.pct_change(21, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["btc_dd_63"] = max_drawdown_from_price(btc, 63).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Simple transparent risk-on score. The gate still applies hard thresholds below.
    out["risk_on_score"] = (
        0.22 * out["btc_above_50"] +
        0.22 * out["btc_above_200"] +
        0.18 * out["eth_above_50"] +
        0.25 * out["breadth_50"].clip(0, 1) +
        0.13 * (out["btc_ret_21"] > 0).astype(float)
    ).clip(0, 1)
    return out.fillna(0.0)


def v26_gate(regime: pd.DataFrame, cfg: dict) -> pd.Series:
    g = cfg["gate"]
    cond = (
        (regime["risk_on_score"] >= float(g["risk_on_score_min"])) &
        (regime["breadth_50"] >= float(g["breadth50_min"])) &
        (regime["btc_dd_63"] >= float(g["min_btc_drawdown_63"]))
    )
    if g.get("require_btc_above_50", True): cond &= regime["btc_above_50"] > 0.5
    if g.get("require_btc_above_200", True): cond &= regime["btc_above_200"] > 0.5
    if g.get("require_eth_above_50", True): cond &= regime["eth_above_50"] > 0.5
    if g.get("require_btc_ret_21_positive", True): cond &= regime["btc_ret_21"] > 0
    confirmation_days = int(g.get("confirmation_days", 1))
    if confirmation_days > 1:
        cond = cond.rolling(confirmation_days).sum() >= confirmation_days
    return cond.fillna(False)


def asset_score_frame(close: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    close = clean_index(close).ffill()
    ret = compute_returns(close)
    lr = np.log1p(ret.clip(lower=-0.999))
    mom21 = lr.rolling(21, min_periods=10).sum()
    mom63 = lr.rolling(63, min_periods=21).sum()
    mom126 = lr.rolling(126, min_periods=42).sum()
    vol63 = ret.rolling(int(cfg["features"].get("volatility_window", 63)), min_periods=21).std() * np.sqrt(252)
    ma50 = close.rolling(50, min_periods=25).mean()
    dd21 = close / close.rolling(int(cfg["features"].get("stop_drawdown_window", 21)), min_periods=10).max() - 1.0
    # Cross-sectional rank score. Higher is better.
    score = 0.30 * mom63.rank(axis=1, pct=True) + 0.20 * mom21.rank(axis=1, pct=True) + 0.20 * mom126.rank(axis=1, pct=True) - 0.15 * vol63.rank(axis=1, pct=True) + 0.15 * (close > ma50).astype(float)
    stop_ok = dd21 >= float(cfg["features"].get("stop_drawdown_threshold", -0.12))
    return score.where(stop_ok).replace([np.inf, -np.inf], np.nan)