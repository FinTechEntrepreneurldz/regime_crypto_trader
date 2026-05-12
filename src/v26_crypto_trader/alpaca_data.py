from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List

import pandas as pd


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper().replace("-", "/")
    if "/" not in symbol and symbol.endswith("USD"):
        symbol = symbol[:-3] + "/USD"
    return symbol


def load_crypto_bars_alpaca(symbols: Iterable[str], lookback_days: int = 430) -> pd.DataFrame:
    """Load daily crypto bars from Alpaca Market Data.

    Returns a DataFrame indexed by timestamp with MultiIndex columns (field, symbol),
    e.g. close['BTC/USD']. Requires APCA_API_KEY_ID and APCA_API_SECRET_KEY.
    """
    try:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except Exception as exc:
        raise ImportError("alpaca-py is required. Install with `pip install alpaca-py`.") from exc

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing APCA_API_KEY_ID/APCA_API_SECRET_KEY environment variables.")

    client = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
    symbols = [normalize_symbol(s) for s in symbols]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    request = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=end)
    bars = client.get_crypto_bars(request)
    df = bars.df.copy()
    if df.empty:
        raise RuntimeError("Alpaca returned no crypto bars. Check symbols and account market-data access.")
    # alpaca-py returns MultiIndex: symbol, timestamp
    if not isinstance(df.index, pd.MultiIndex):
        raise RuntimeError("Unexpected Alpaca bars format; expected MultiIndex(symbol, timestamp).")
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    field_map = {}
    for field in ["open", "high", "low", "close", "volume"]:
        piv = df.pivot(index="timestamp", columns="symbol", values=field).sort_index()
        field_map[field] = piv
    out = pd.concat(field_map, axis=1)
    return out


def bars_to_close_volume(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(bars.columns, pd.MultiIndex):
        close = bars["close"].copy()
        volume = bars["volume"].copy() if "volume" in bars.columns.get_level_values(0) else close * 0.0
    else:
        raise ValueError("Expected MultiIndex columns from load_crypto_bars_alpaca")
    close = close.sort_index().ffill()
    volume = volume.sort_index().fillna(0.0)
    return close, volume