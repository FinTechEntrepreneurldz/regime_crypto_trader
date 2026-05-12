from __future__ import annotations

from pathlib import Path
import pandas as pd


def snapshot_performance(equity: float, positions: pd.DataFrame, signal: dict) -> pd.DataFrame:
    gross = 0.0 if positions.empty else positions["market_value"].abs().sum() / float(equity)
    return pd.DataFrame([{
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "equity": equity,
        "gross_exposure": gross,
        "active_signal": signal.get("active"),
        "risk_on_score": signal.get("risk_on_score"),
        "n_positions": 0 if positions.empty else len(positions),
    }])