from __future__ import annotations

import os
import pandas as pd

from .settings import first_env


def normalize_alpaca_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "")
    if s.endswith("USD"):
        return s[:-3] + "/USD"
    return symbol.upper()


def trading_client():
    try:
        from alpaca.trading.client import TradingClient
    except Exception as exc:
        raise ImportError("alpaca-py is required. Install with `pip install alpaca-py`.") from exc

    key = first_env("APCA_API_KEY_ID", "ALPACA_API_KEY", "ALPACA_REGIME_CRYPTO_KEY_ID")
    secret = first_env("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY", "ALPACA_REGIME_CRYPTO_SECRET_KEY")

    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Add APCA_API_KEY_ID/APCA_API_SECRET_KEY or "
            "ALPACA_API_KEY/ALPACA_SECRET_KEY as GitHub secrets."
        )

    return TradingClient(key, secret, paper=True)


def get_account_info(client=None) -> dict:
    client = client or trading_client()
    account = client.get_account()
    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "long_market_value": float(account.long_market_value),
        "short_market_value": float(account.short_market_value),
        "portfolio_value": float(account.portfolio_value),
        "status": str(account.status),
    }


def get_account_equity(client=None) -> float:
    return float(get_account_info(client).get("equity", 0.0))


def get_positions(client=None) -> pd.DataFrame:
    client = client or trading_client()
    positions = client.get_all_positions()
    rows = []
    for p in positions:
        symbol = normalize_alpaca_symbol(str(p.symbol))
        qty = float(p.qty)
        market_value = float(p.market_value)
        rows.append({
            "symbol_raw": str(p.symbol),
            "symbol": symbol,
            "qty": qty,
            "market_value": market_value,
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl),
            "asset_class": str(getattr(p, "asset_class", "")).lower(),
            "side": "long" if qty >= 0 else "short",
        })
    return pd.DataFrame(rows)


def submit_market_order(client, symbol: str, side: str, notional: float | None = None, qty: float | None = None):
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    kwargs = {"symbol": symbol.upper(), "side": side_enum, "time_in_force": TimeInForce.GTC}

    if notional is not None:
        kwargs["notional"] = round(float(notional), 2)
    elif qty is not None:
        kwargs["qty"] = float(qty)
    else:
        raise ValueError("Either notional or qty must be supplied.")

    return client.submit_order(MarketOrderRequest(**kwargs))
