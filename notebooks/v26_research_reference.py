# -*- coding: utf-8 -*-
# Exported from v26_risk_on_only_crypto_relval_production_colab.ipynb

# %% [markdown]
# # V26 Risk-On Only Crypto Relative Value Production Lab
# 
# This notebook resets the crypto strategy research process around **exchange-native crypto data** and **market-neutral / relative-value engines**.
# 
# It intentionally avoids PPO, MLP, LLM, equity alpha, and Yahoo-only research loops. V26 keeps the V25 exchange-native engines, but the final strategy trades only when the crypto regime is confirmed risk-on and stays flat everywhere else.
# 
# Core engines:
# 
# 1. Funding carry
# 2. Basis carry
# 3. Pairs / stat-arb
# 4. BTC/ETH beta-neutral construction
# 5. Stop-loss momentum
# 6. Liquidity-aware reversal
# 
# The notebook uses CCXT first. If real funding/basis/open-interest data is unavailable, the affected engines are disabled rather than faked.
# 
# ## V26 design
# 
# The final selector is risk-on-only: it builds candidate strategies from the V25 relative-value streams, gates them with causal prior-day regime features, validates them across folds, and reports the untouched final test period separately.

# %%
# ============================================================
# 0. Install dependencies, mount Drive, imports
# ============================================================
import sys, subprocess, importlib.util, warnings, os, json, math, time, gc
from pathlib import Path
from collections import OrderedDict, defaultdict


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing {pip_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])
    else:
        print(f"{pip_name} already installed.")

for import_name, pip_name in [
    ("ccxt", "ccxt"),
    ("pyarrow", "pyarrow"),
    ("openpyxl", "openpyxl"),
    ("statsmodels", "statsmodels"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
]:
    ensure_package(import_name, pip_name)

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as exc:
    print('Google Drive mount skipped:', repr(exc))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import ccxt
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 220)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print('Imports ready.')

# %%
# ============================================================
# 1. Configuration
# ============================================================
DRIVE_ROOT = Path('/content/drive/MyDrive')
DEFAULT_OUTDIR = DRIVE_ROOT / 'V26_RISK_ON_ONLY_CRYPTO_RELVAL_PRODUCTION' if DRIVE_ROOT.exists() else Path('/mnt/data/V26_RISK_ON_ONLY_CRYPTO_RELVAL_PRODUCTION')

CONFIG = {
    'NAME': 'V26_RISK_ON_ONLY_CRYPTO_RELVAL_PRODUCTION',
    'OUTDIR': str(DEFAULT_OUTDIR),

    # Exchange search order. Public endpoints only.
    # binanceusdm can be geo-restricted; bybit/okx often work better in Colab.
    'EXCHANGE_IDS': ['bybit', 'okx', 'binanceusdm'],
    'QUOTE': 'USDT',
    'PERP_SETTLE': 'USDT',
    'PREFERRED_MARKET_TYPE': 'swap',

    # Data range. More history improves fold testing, but CCXT public history can be slow.
    'START': '2020-01-01',
    'END': None,

    # Split dates.
    'TRAIN_START': '2020-01-01',
    'TRAIN_END': '2023-12-31',
    'VALIDATION_START': '2024-01-01',
    'VALIDATION_END': '2024-12-31',
    'TEST_START': '2025-01-01',
    'TEST_END': None,

    # Universe.
    'TOP_N_PERPS': 120,
    'MIN_HISTORY_DAYS': 540,
    'MIN_DAILY_DOLLAR_VOLUME': 2_000_000,
    'MAX_ASSETS_FOR_PAIR_SEARCH': 35,
    'MAX_ASSETS_FOR_DAILY_LS': 80,
    'EXCLUDE_BASES': ['USDT','USDC','BUSD','FDUSD','DAI','TUSD','USDD','UST','USTC','EUR','USD','TRY','BRL','GBP','JPY','AUD'],
    'EXCLUDE_SUBSTRINGS': ['UP/', 'DOWN/', 'BULL/', 'BEAR/', '3L/', '3S/', '5L/', '5S/'],

    # Data fetch controls.
    'FETCH_DAILY_OHLCV': True,
    'FETCH_4H_OHLCV': True,
    'FETCH_SPOT_OHLCV': True,
    'FETCH_FUNDING': True,
    'FETCH_OPEN_INTEREST': True,
    'OHLCV_LIMIT_PER_CALL': 1000,
    'SLEEP_SECONDS': 0.08,
    'REFRESH_CACHE': False,

    # Trading / execution assumptions.
    'REBALANCE_FREQ': 'D',
    'BASE_TRANSACTION_COST_BPS': 8.0,
    'SHORT_FEE_ANNUAL': 0.08,
    'MAX_SINGLE_NAME_WEIGHT': 0.08,
    'LONG_GROSS': 0.65,
    'SHORT_GROSS': 0.65,
    'MAX_TOTAL_GROSS': 1.30,
    'PARTIAL_ADJUSTMENT_ALPHA': 0.50,

    # Risk controls.
    'TARGET_VOL': 0.20,
    'MAX_LEVERAGE': 1.25,
    'DOWNSIDE_TARGET_VOL': 0.18,
    'STOP_LOSS_5D': -0.06,
    'STOP_LOSS_10D': -0.10,
    'TRAILING_DD_21D': -0.14,
    'PORTFOLIO_DD_CUT': -0.12,
    'PORTFOLIO_DD_SCALE': 0.35,

    # Signal params.
    'MOM_FAST': 7,
    'MOM_MED': 21,
    'MOM_SLOW': 63,
    'REV_FAST': 1,
    'LIQUID_CORE_N': 20,
    'LONG_N': 8,
    'SHORT_N': 8,
    'STRICT_SHORT_REQUIRE_FUNDING_OR_BASIS': True,

    # Pair stat-arb.
    'PAIR_LOOKBACK': 180,
    'PAIR_REFRESH_DAYS': 7,
    'PAIR_MIN_CORR': 0.65,
    'PAIR_MAX_COINTEGRATION_P': 0.08,
    'PAIR_MIN_HALFLIFE': 2.0,
    'PAIR_MAX_HALFLIFE': 35.0,
    'PAIR_Z_ENTRY': 1.75,
    'PAIR_Z_EXIT': 0.25,
    'PAIR_Z_STOP': 3.25,
    'MAX_ACTIVE_PAIRS': 10,

    # Carry.
    'FUNDING_MIN_ABS_DAILY': 0.00002,
    'BASIS_MIN_ABS': 0.0005,
    'CARRY_TOP_N': 8,

    # Fold reporting.
    'FOLD_START': '2021-01-01',
    'FOLD_DAYS': 180,
    'FOLD_STEP_DAYS': 90,

    # Selector.
    'REQUIRE_POSITIVE_VALIDATION': True,
    'MIN_VALIDATION_SHARPE': 0.25,

    # V26 risk-on-only final production selector.
    'V26_RISK_ON_SCORE_GRID': [0.65, 0.70, 0.75, 0.80],
    'V26_BREADTH_50_GRID': [0.45, 0.50, 0.55, 0.60],
    'V26_CONFIRMATION_DAYS_GRID': [0, 2, 5],
    'V26_TRAIL_WINDOWS': [0, 21, 63],
    'V26_TRAIL_SHARPE_MIN_GRID': [-0.25, 0.00, 0.25],
    'V26_TARGET_VOL_GRID': [None, 0.08, 0.12, 0.16, 0.20],
    'V26_MAX_LEVERAGE': 1.50,
    'V26_MIN_ACTIVE_DAYS_VALIDATION': 20,
    'V26_MIN_ACTIVE_RATE_VALIDATION': 0.05,
    'V26_MIN_MEDIAN_FOLD_SHARPE': 0.25,
    'V26_REQUIRE_POSITIVE_TEST': False,
}

OUTDIR = Path(CONFIG['OUTDIR'])
CACHE_DIR = OUTDIR / 'cache'
OUTDIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(json.dumps(CONFIG, indent=2, default=str))

# %%
# ============================================================
# 2. Generic helpers
# ============================================================
def to_ms(dt):
    return int(pd.Timestamp(dt, tz='UTC').timestamp() * 1000)


def clean_dt_index(idx):
    idx = pd.to_datetime(idx, utc=True).tz_convert(None)
    return idx


def series(x, name=None):
    s = pd.Series(x).copy()
    s.index = clean_dt_index(s.index)
    s = s.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if name is not None:
        s.name = name
    return s


def compound(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return 0.0
    return float((1.0 + x).prod() - 1.0)


def ann_sharpe(x, periods=365):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 20 or x.std() <= 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(periods))


def max_drawdown(x):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return np.nan
    eq = (1.0 + x).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def perf_stats(x, start=None, end=None, periods=365):
    s = series(x)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
    s = s.dropna()
    if len(s) < 20:
        return dict(ann_return=np.nan, ann_vol=np.nan, sharpe=np.nan, sortino=np.nan, tstat=np.nan,
                    max_dd=np.nan, calmar=np.nan, total_return=np.nan, hit_rate=np.nan, n_days=len(s))
    eq = (1.0 + s).cumprod()
    ann_return = eq.iloc[-1] ** (periods / len(s)) - 1.0
    ann_vol = s.std() * np.sqrt(periods)
    sharpe = s.mean() / s.std() * np.sqrt(periods) if s.std() > 0 else np.nan
    downside = s[s < 0].std() * np.sqrt(periods)
    sortino = s.mean() * periods / downside if pd.notna(downside) and downside > 0 else np.nan
    tstat = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else np.nan
    mdd = max_drawdown(s)
    calmar = ann_return / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan
    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe, sortino=sortino, tstat=tstat,
                max_dd=mdd, calmar=calmar, total_return=eq.iloc[-1] - 1.0,
                hit_rate=float((s > 0).mean()), n_days=len(s))


def robust_z(x, clip=5.0):
    x = pd.Series(x, dtype=float).replace([np.inf, -np.inf], np.nan)
    med = x.median()
    mad = (x - med).abs().median()
    if pd.isna(mad) or mad <= 1e-12:
        sd = x.std()
        if pd.isna(sd) or sd <= 1e-12:
            return pd.Series(0.0, index=x.index)
        z = (x - x.mean()) / sd
    else:
        z = 0.6745 * (x - med) / mad
    return z.clip(-clip, clip).fillna(0.0)


def cap_weights(w, cap=None, target=1.0):
    cap = CONFIG['MAX_SINGLE_NAME_WEIGHT'] if cap is None else cap
    w = pd.Series(w, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if target <= 0 or w.sum() <= 0:
        return w * 0.0
    w = w / w.sum() * target
    for _ in range(30):
        over = w > cap + 1e-12
        if not over.any():
            break
        fixed = w[over].clip(upper=cap)
        free = w[~over]
        rem = target - fixed.sum()
        if rem <= 0 or free.empty or free.sum() <= 0:
            out = pd.concat([fixed, free * 0.0]).reindex(w.index).fillna(0.0)
            return out
        free = free / free.sum() * rem
        w = pd.concat([fixed, free]).reindex(w.index).fillna(0.0)
    return w


def inverse_vol_weights(cols, ret_df, dt, gross=1.0, lookback=30, cap=None):
    cols = [c for c in cols if c in ret_df.columns]
    out = pd.Series(0.0, index=ret_df.columns)
    if not cols:
        return out
    hist = ret_df.loc[:pd.Timestamp(dt), cols].tail(lookback)
    vol = hist.std().replace(0, np.nan)
    inv = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if inv.sum() <= 0:
        inv = pd.Series(1.0, index=cols)
    w = cap_weights(inv, cap=cap, target=gross)
    out.loc[w.index] = w.values
    return out


def apply_vol_target(ret, target_vol=0.20, lookback=30, max_leverage=1.25, periods=365):
    r = series(ret)
    vol = r.rolling(lookback, min_periods=max(10, lookback // 2)).std().shift(1) * np.sqrt(periods)
    scale = (target_vol / vol).replace([np.inf, -np.inf], np.nan).clip(0.0, max_leverage).fillna(0.0)
    return (r * scale).rename(getattr(ret, 'name', None))


def apply_downside_vol_target(ret, target_vol=0.18, lookback=30, max_leverage=1.25, periods=365):
    r = series(ret)
    neg = r.where(r < 0, 0.0)
    dvol = neg.rolling(lookback, min_periods=max(10, lookback // 2)).std().shift(1) * np.sqrt(periods)
    scale = (target_vol / dvol).replace([np.inf, -np.inf], np.nan).clip(0.0, max_leverage).fillna(0.0)
    return (r * scale).rename(getattr(ret, 'name', None))


def plot_equity_curve(series_dict, title, start=None, end=None):
    plt.figure(figsize=(13, 5))
    for name, ret in series_dict.items():
        s = series(ret)
        if start is not None:
            s = s[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s[s.index <= pd.Timestamp(end)]
        eq = (1.0 + s.fillna(0.0)).cumprod()
        plt.plot(eq.index, eq.values, label=str(name), linewidth=1.25)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.show()

# %%
# ============================================================
# 3. Exchange selection and market universe
# ============================================================
def make_exchange(exchange_id, market_type='swap'):
    cls = getattr(ccxt, exchange_id)
    ex = cls({'enableRateLimit': True, 'timeout': 30000})
    # Many exchanges use defaultType; some ignore it harmlessly.
    ex.options = dict(getattr(ex, 'options', {}) or {})
    ex.options['defaultType'] = market_type
    return ex


def is_bad_symbol(symbol, market):
    base = str(market.get('base', '')).upper()
    if base in CONFIG['EXCLUDE_BASES']:
        return True
    s = str(symbol).upper()
    if any(bad in s for bad in CONFIG['EXCLUDE_SUBSTRINGS']):
        return True
    return False


def get_candidate_perp_markets(ex):
    ex.load_markets()
    markets = []
    for sym, m in ex.markets.items():
        try:
            if not m.get('active', True):
                continue
            if not (m.get('swap') or m.get('future')):
                continue
            if str(m.get('quote', '')).upper() != CONFIG['QUOTE']:
                continue
            if str(m.get('settle', CONFIG['PERP_SETTLE'])).upper() != CONFIG['PERP_SETTLE']:
                continue
            if is_bad_symbol(sym, m):
                continue
            markets.append(sym)
        except Exception:
            continue
    return markets


def get_candidate_spot_markets(ex):
    ex.load_markets()
    spot_by_base = {}
    for sym, m in ex.markets.items():
        try:
            if not m.get('active', True) or not m.get('spot', False):
                continue
            if str(m.get('quote', '')).upper() != CONFIG['QUOTE']:
                continue
            if is_bad_symbol(sym, m):
                continue
            base = str(m.get('base', '')).upper()
            spot_by_base[base] = sym
        except Exception:
            continue
    return spot_by_base


def ticker_quote_volume(ticker):
    for k in ['quoteVolume', 'quoteVolume24h', 'turnover', 'baseVolume']:
        v = ticker.get(k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    info = ticker.get('info', {}) or {}
    for k in ['quoteVolume', 'turnover24h', 'turnover', 'volume24h']:
        v = info.get(k, None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return np.nan


def select_exchange_and_universe():
    attempts = []
    for exchange_id in CONFIG['EXCHANGE_IDS']:
        try:
            ex = make_exchange(exchange_id, 'swap')
            symbols = get_candidate_perp_markets(ex)
            print(exchange_id, 'candidate perp symbols:', len(symbols))
            if len(symbols) < 20:
                attempts.append({'exchange_id': exchange_id, 'status': 'too_few_symbols', 'n': len(symbols)})
                continue
            # Fetch tickers in batch if possible. Fall back to per symbol.
            tickers = {}
            try:
                tickers = ex.fetch_tickers(symbols)
            except Exception as exc:
                print(exchange_id, 'fetch_tickers batch failed, sampling individually:', repr(exc))
                for sym in tqdm(symbols[:min(250, len(symbols))], desc=f'{exchange_id} tickers'):
                    try:
                        tickers[sym] = ex.fetch_ticker(sym)
                        time.sleep(float(CONFIG['SLEEP_SECONDS']))
                    except Exception:
                        pass
            rows = []
            for sym in symbols:
                m = ex.markets.get(sym, {})
                base = str(m.get('base', '')).upper()
                qv = ticker_quote_volume(tickers.get(sym, {})) if sym in tickers else np.nan
                rows.append({'exchange_id': exchange_id, 'symbol': sym, 'base': base, 'quote_volume': qv})
            uni = pd.DataFrame(rows)
            uni = uni.sort_values('quote_volume', ascending=False, na_position='last')
            if uni['quote_volume'].notna().sum() > 20:
                uni = uni[uni['quote_volume'].fillna(0.0) >= float(CONFIG['MIN_DAILY_DOLLAR_VOLUME'])]
            uni = uni.head(int(CONFIG['TOP_N_PERPS'])).reset_index(drop=True)
            if len(uni) >= 10:
                print('Selected exchange:', exchange_id, 'universe:', len(uni))
                return ex, exchange_id, uni
            attempts.append({'exchange_id': exchange_id, 'status': 'filtered_too_small', 'n': len(uni)})
        except Exception as exc:
            attempts.append({'exchange_id': exchange_id, 'status': 'failed', 'error': repr(exc)})
            print(exchange_id, 'failed:', repr(exc))
    raise RuntimeError(f'No exchange produced a usable perp universe. Attempts={attempts}')

EXCHANGE, EXCHANGE_ID, UNIVERSE = select_exchange_and_universe()
ASSETS = UNIVERSE['symbol'].tolist()
BASE_BY_SYMBOL = dict(zip(UNIVERSE['symbol'], UNIVERSE['base']))
print('Selected exchange:', EXCHANGE_ID)
display(UNIVERSE.head(40))

# %%
# ============================================================
# 4. Exchange-native data loaders with cache
# ============================================================
def cache_key_symbol(symbol):
    return str(symbol).replace('/', '_').replace(':', '_').replace('-', '_')


def fetch_ohlcv_paginated(ex, symbol, timeframe, since_ms, end_ms=None, limit=1000):
    all_rows = []
    cursor = since_ms
    loops = 0
    while True:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except Exception as exc:
            if loops == 0:
                raise
            print('OHLCV fetch stopped for', symbol, timeframe, repr(exc))
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        next_cursor = last_ts + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        loops += 1
        if end_ms is not None and last_ts >= end_ms:
            break
        if len(rows) < min(100, limit):
            break
        time.sleep(float(CONFIG['SLEEP_SECONDS']))
    if not all_rows:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume']).drop_duplicates('timestamp')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert(None)
    df = df.set_index('datetime').sort_index()[['open','high','low','close','volume']]
    if end_ms is not None:
        end_dt = pd.to_datetime(end_ms, unit='ms', utc=True).tz_convert(None)
        df = df[df.index <= end_dt]
    return df.astype(float)


def fetch_funding_history(ex, symbol, since_ms, end_ms=None):
    rows_all = []
    if not hasattr(ex, 'fetch_funding_rate_history'):
        return pd.DataFrame()
    cursor = since_ms
    for _ in range(200):
        try:
            rows = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        except Exception as exc:
            # Some exchanges require different symbols/endpoints.
            if not rows_all:
                print('Funding unavailable for', symbol, repr(exc))
            break
        if not rows:
            break
        rows_all.extend(rows)
        timestamps = [r.get('timestamp') for r in rows if r.get('timestamp') is not None]
        if not timestamps:
            break
        last_ts = max(timestamps)
        cursor = last_ts + 1
        if end_ms is not None and last_ts >= end_ms:
            break
        time.sleep(float(CONFIG['SLEEP_SECONDS']))
    if not rows_all:
        return pd.DataFrame()
    out = []
    for r in rows_all:
        ts = r.get('timestamp')
        rate = r.get('fundingRate', r.get('rate', None))
        if ts is None or rate is None:
            continue
        out.append({'datetime': pd.to_datetime(ts, unit='ms', utc=True).tz_convert(None), 'funding_rate': float(rate)})
    df = pd.DataFrame(out).drop_duplicates('datetime').set_index('datetime').sort_index() if out else pd.DataFrame()
    return df


def fetch_open_interest_history_safe(ex, symbol, since_ms, end_ms=None, timeframe='1d'):
    rows_all = []
    if not hasattr(ex, 'fetch_open_interest_history'):
        return pd.DataFrame()
    cursor = since_ms
    for _ in range(100):
        try:
            rows = ex.fetch_open_interest_history(symbol, timeframe=timeframe, since=cursor, limit=500)
        except Exception as exc:
            if not rows_all:
                print('Open interest unavailable for', symbol, repr(exc))
            break
        if not rows:
            break
        rows_all.extend(rows)
        timestamps = [r.get('timestamp') for r in rows if r.get('timestamp') is not None]
        if not timestamps:
            break
        last_ts = max(timestamps)
        cursor = last_ts + 1
        if end_ms is not None and last_ts >= end_ms:
            break
        time.sleep(float(CONFIG['SLEEP_SECONDS']))
    if not rows_all:
        return pd.DataFrame()
    out = []
    for r in rows_all:
        ts = r.get('timestamp')
        oi = r.get('openInterestAmount', r.get('openInterestValue', r.get('openInterest', None)))
        if ts is None or oi is None:
            continue
        try:
            oi = float(oi)
        except Exception:
            continue
        out.append({'datetime': pd.to_datetime(ts, unit='ms', utc=True).tz_convert(None), 'open_interest': oi})
    df = pd.DataFrame(out).drop_duplicates('datetime').set_index('datetime').sort_index() if out else pd.DataFrame()
    return df


def load_or_fetch_ohlcv(symbol, timeframe='1d', market_type='swap'):
    fname = CACHE_DIR / f'{EXCHANGE_ID}_{market_type}_{cache_key_symbol(symbol)}_{timeframe}_ohlcv.parquet'
    if fname.exists() and not CONFIG['REFRESH_CACHE']:
        return pd.read_parquet(fname)
    ex = make_exchange(EXCHANGE_ID, market_type)
    ex.load_markets()
    df = fetch_ohlcv_paginated(ex, symbol, timeframe, to_ms(CONFIG['START']), to_ms(CONFIG['END']) if CONFIG['END'] else None, int(CONFIG['OHLCV_LIMIT_PER_CALL']))
    if not df.empty:
        df.to_parquet(fname)
    return df


def load_or_fetch_funding(symbol):
    fname = CACHE_DIR / f'{EXCHANGE_ID}_{cache_key_symbol(symbol)}_funding.parquet'
    if fname.exists() and not CONFIG['REFRESH_CACHE']:
        return pd.read_parquet(fname)
    ex = make_exchange(EXCHANGE_ID, 'swap')
    ex.load_markets()
    df = fetch_funding_history(ex, symbol, to_ms(CONFIG['START']), to_ms(CONFIG['END']) if CONFIG['END'] else None)
    if not df.empty:
        df.to_parquet(fname)
    return df


def load_or_fetch_open_interest(symbol):
    fname = CACHE_DIR / f'{EXCHANGE_ID}_{cache_key_symbol(symbol)}_open_interest.parquet'
    if fname.exists() and not CONFIG['REFRESH_CACHE']:
        return pd.read_parquet(fname)
    ex = make_exchange(EXCHANGE_ID, 'swap')
    ex.load_markets()
    df = fetch_open_interest_history_safe(ex, symbol, to_ms(CONFIG['START']), to_ms(CONFIG['END']) if CONFIG['END'] else None, timeframe='1d')
    if not df.empty:
        df.to_parquet(fname)
    return df


def find_spot_symbol_for_base(base):
    try:
        spot_ex = make_exchange(EXCHANGE_ID, 'spot')
        spot_by_base = get_candidate_spot_markets(spot_ex)
        return spot_by_base.get(str(base).upper())
    except Exception:
        return None

print('Data loader helpers ready.')

# %%
# ============================================================
# 5. Fetch data and build panels
# ============================================================
perp_close, perp_volume = {}, {}
perp_ohlcv_daily = {}
perp_ohlcv_4h = {}
funding_daily = {}
open_interest_daily = {}
spot_close = {}
fetch_rows = []

# Precompute spot map once.
try:
    SPOT_EXCHANGE = make_exchange(EXCHANGE_ID, 'spot')
    SPOT_BY_BASE = get_candidate_spot_markets(SPOT_EXCHANGE)
except Exception as exc:
    print('Spot market lookup failed:', repr(exc))
    SPOT_BY_BASE = {}

for sym in tqdm(ASSETS, desc='Fetching exchange-native data'):
    base = BASE_BY_SYMBOL.get(sym, '')
    row = {'symbol': sym, 'base': base}
    try:
        df = load_or_fetch_ohlcv(sym, '1d', 'swap')
        if df.empty:
            row['perp_daily_status'] = 'empty'
            fetch_rows.append(row)
            continue
        perp_ohlcv_daily[sym] = df
        perp_close[sym] = df['close']
        perp_volume[sym] = df['volume']
        row['perp_daily_rows'] = len(df)
    except Exception as exc:
        row['perp_daily_error'] = repr(exc)
        fetch_rows.append(row)
        continue

    if CONFIG['FETCH_4H_OHLCV']:
        try:
            df4 = load_or_fetch_ohlcv(sym, '4h', 'swap')
            if not df4.empty:
                perp_ohlcv_4h[sym] = df4
                row['perp_4h_rows'] = len(df4)
        except Exception as exc:
            row['perp_4h_error'] = repr(exc)

    if CONFIG['FETCH_FUNDING']:
        try:
            fd = load_or_fetch_funding(sym)
            if not fd.empty:
                funding_daily[sym] = fd['funding_rate'].resample('1D').sum()
                row['funding_rows'] = len(fd)
        except Exception as exc:
            row['funding_error'] = repr(exc)

    if CONFIG['FETCH_OPEN_INTEREST']:
        try:
            oi = load_or_fetch_open_interest(sym)
            if not oi.empty:
                open_interest_daily[sym] = oi['open_interest'].resample('1D').last()
                row['oi_rows'] = len(oi)
        except Exception as exc:
            row['oi_error'] = repr(exc)

    if CONFIG['FETCH_SPOT_OHLCV']:
        spot_sym = SPOT_BY_BASE.get(str(base).upper())
        row['spot_symbol'] = spot_sym
        if spot_sym:
            try:
                sdf = load_or_fetch_ohlcv(spot_sym, '1d', 'spot')
                if not sdf.empty:
                    spot_close[sym] = sdf['close']
                    row['spot_rows'] = len(sdf)
            except Exception as exc:
                row['spot_error'] = repr(exc)
    fetch_rows.append(row)
    gc.collect()

FETCH_REPORT = pd.DataFrame(fetch_rows)
print('Fetch report:')
display(FETCH_REPORT.head(30))

close = pd.DataFrame(perp_close).sort_index()
volume = pd.DataFrame(perp_volume).reindex(close.index)
funding = pd.DataFrame(funding_daily).reindex(close.index).fillna(0.0)
open_interest = pd.DataFrame(open_interest_daily).reindex(close.index).ffill()
spot = pd.DataFrame(spot_close).reindex(close.index).ffill()

# Basic data filters.
coverage = close.notna().mean()
history_days = close.notna().sum()
vol_usd_proxy = (close * volume).rolling(30, min_periods=10).mean().median()
keep = coverage.index[(history_days >= int(CONFIG['MIN_HISTORY_DAYS'])) & (vol_usd_proxy.reindex(coverage.index).fillna(0) >= float(CONFIG['MIN_DAILY_DOLLAR_VOLUME']))].tolist()
if len(keep) < 10:
    # Loosen volume if the exchange data lacks reliable volume field.
    keep = coverage.index[(history_days >= int(CONFIG['MIN_HISTORY_DAYS']))].tolist()
if len(keep) < 6:
    raise RuntimeError(f'Too few assets survived history filters: {len(keep)}')

keep = keep[:int(CONFIG['TOP_N_PERPS'])]
close = close[keep].ffill().dropna(how='all')
volume = volume.reindex(close.index)[keep].fillna(0.0)
funding = funding.reindex(close.index).reindex(columns=keep).fillna(0.0)
open_interest = open_interest.reindex(close.index).reindex(columns=keep).ffill()
spot = spot.reindex(close.index).reindex(columns=keep).ffill()
ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.50, 0.50)

# Basis: perp spot premium. Only available when spot close exists.
basis = (close / spot - 1.0).replace([np.inf, -np.inf], np.nan)
basis_change = basis.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

ASSETS = close.columns.tolist()
BASES = [BASE_BY_SYMBOL.get(s, s.split('/')[0]) for s in ASSETS]
print('Final asset count:', len(ASSETS))
print('Date range:', close.index.min(), 'to', close.index.max())
print('Funding nonzero mean abs:', float(funding.abs().mean().mean()))
print('Basis coverage:', float(basis.notna().mean().mean()))
print('Open interest coverage:', float(open_interest.notna().mean().mean()))
display(pd.DataFrame({'asset': ASSETS, 'base': BASES, 'coverage': coverage.reindex(ASSETS).values, 'hist_days': history_days.reindex(ASSETS).values, 'dollar_vol_proxy': vol_usd_proxy.reindex(ASSETS).values}).sort_values('dollar_vol_proxy', ascending=False).head(50))

# %%
# ============================================================
# 6. Intraday 4h features, regime, liquidity ranks
# ============================================================
# Build 4h features aggregated to daily. Missing 4h data becomes zeros.
features_4h = {}
for sym, df4 in perp_ohlcv_4h.items():
    if sym not in ASSETS or df4.empty:
        continue
    r4 = df4['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    daily = pd.DataFrame({
        'last_4h_ret': r4.resample('1D').last(),
        'sum_4h_ret': r4.resample('1D').sum(),
        'intraday_vol': r4.resample('1D').std() * np.sqrt(6),
        'jump_count': (r4.abs() > 0.035).astype(float).resample('1D').sum(),
        'intraday_volume': df4['volume'].resample('1D').sum(),
    })
    features_4h[sym] = daily

intraday_last4h = pd.DataFrame({s: d['last_4h_ret'] for s, d in features_4h.items()}).reindex(close.index).reindex(columns=ASSETS).fillna(0.0)
intraday_sum = pd.DataFrame({s: d['sum_4h_ret'] for s, d in features_4h.items()}).reindex(close.index).reindex(columns=ASSETS).fillna(0.0)
intraday_vol = pd.DataFrame({s: d['intraday_vol'] for s, d in features_4h.items()}).reindex(close.index).reindex(columns=ASSETS).fillna(0.0)
intraday_jump_count = pd.DataFrame({s: d['jump_count'] for s, d in features_4h.items()}).reindex(close.index).reindex(columns=ASSETS).fillna(0.0)

# Liquidity ranks by rolling dollar volume proxy.
dollar_vol = (close * volume).replace([np.inf, -np.inf], np.nan)
adv30 = dollar_vol.rolling(30, min_periods=10).mean()
liq_rank = adv30.rank(axis=1, ascending=False, method='first')

# BTC/ETH proxies.
def find_asset_by_base(base):
    base = base.upper()
    for s in ASSETS:
        if BASE_BY_SYMBOL.get(s, '').upper() == base or s.upper().startswith(base + '/'):
            return s
    return None

BTC_ASSET = find_asset_by_base('BTC') or ASSETS[0]
ETH_ASSET = find_asset_by_base('ETH') or (ASSETS[1] if len(ASSETS) > 1 else ASSETS[0])
print('BTC proxy:', BTC_ASSET, 'ETH proxy:', ETH_ASSET)

btc_px = close[BTC_ASSET].ffill()
eth_px = close[ETH_ASSET].ffill()
market_ret = ret[[c for c in [BTC_ASSET, ETH_ASSET] if c in ret.columns]].mean(axis=1)

regime = pd.DataFrame(index=close.index)
regime['btc_ret_21'] = ret[BTC_ASSET].rolling(21, min_periods=10).sum()
regime['btc_ret_63'] = ret[BTC_ASSET].rolling(63, min_periods=20).sum()
regime['btc_above_50'] = (btc_px > btc_px.rolling(50, min_periods=20).mean()).astype(float)
regime['btc_above_200'] = (btc_px > btc_px.rolling(200, min_periods=80).mean()).astype(float)
regime['eth_above_50'] = (eth_px > eth_px.rolling(50, min_periods=20).mean()).astype(float)
regime['breadth_50'] = (close > close.rolling(50, min_periods=20).mean()).mean(axis=1)
regime['breadth_200'] = (close > close.rolling(200, min_periods=80).mean()).mean(axis=1)
regime['market_vol_21'] = market_ret.rolling(21, min_periods=10).std() * np.sqrt(365)
eq_btc = (1 + ret[BTC_ASSET]).cumprod()
regime['btc_dd_63'] = eq_btc / eq_btc.rolling(63, min_periods=20).max() - 1.0
regime['risk_on_score'] = (
    0.25 * regime['btc_above_50'] +
    0.20 * regime['btc_above_200'] +
    0.15 * regime['eth_above_50'] +
    0.25 * regime['breadth_50'] +
    0.15 * (regime['btc_ret_21'] > 0).astype(float)
).clip(0, 1)
regime['regime'] = np.select(
    [regime['risk_on_score'] >= 0.65, regime['risk_on_score'] <= 0.35],
    ['risk_on', 'risk_off'],
    default='neutral'
)
regime = regime.replace([np.inf, -np.inf], np.nan).fillna(0.0)
display(regime.tail())

# %%
# ============================================================
# 7. BTC/ETH beta estimation and beta-neutral construction
# ============================================================
def rolling_beta_to_factor(asset_returns, factor_returns, lookback=60):
    cov = asset_returns.rolling(lookback, min_periods=25).cov(factor_returns)
    var = factor_returns.rolling(lookback, min_periods=25).var().replace(0, np.nan)
    return (cov / var).replace([np.inf, -np.inf], np.nan)

beta_btc = pd.DataFrame({c: rolling_beta_to_factor(ret[c], ret[BTC_ASSET]) for c in ASSETS}).reindex(close.index)
beta_eth = pd.DataFrame({c: rolling_beta_to_factor(ret[c], ret[ETH_ASSET]) for c in ASSETS}).reindex(close.index)


def normalize_long_short(w, long_gross=None, short_gross=None, max_total_gross=None):
    w = pd.Series(w, index=ASSETS, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    long_gross = CONFIG['LONG_GROSS'] if long_gross is None else long_gross
    short_gross = CONFIG['SHORT_GROSS'] if short_gross is None else short_gross
    max_total_gross = CONFIG['MAX_TOTAL_GROSS'] if max_total_gross is None else max_total_gross
    pos = w.clip(lower=0.0)
    neg = (-w.clip(upper=0.0))
    if pos.sum() > 0:
        pos = cap_weights(pos, target=long_gross)
    if neg.sum() > 0:
        neg = cap_weights(neg, target=short_gross)
    out = pos - neg
    gross = out.abs().sum()
    if gross > max_total_gross:
        out *= max_total_gross / gross
    return out.reindex(ASSETS).fillna(0.0)


def beta_neutralize(w, dt, target_btc=0.0, target_eth=0.0, max_iter=3):
    w = pd.Series(w, index=ASSETS, dtype=float).fillna(0.0)
    dt = pd.Timestamp(dt)
    for _ in range(max_iter):
        b1 = beta_btc.loc[:dt].iloc[-1].reindex(ASSETS).fillna(0.0)
        b2 = beta_eth.loc[:dt].iloc[-1].reindex(ASSETS).fillna(0.0)
        B = np.vstack([b1.values, b2.values])
        exposure = B @ w.values - np.array([target_btc, target_eth])
        if np.linalg.norm(exposure) < 1e-4:
            break
        # Solve minimum-norm weight adjustment using beta matrix.
        try:
            adj = B.T @ np.linalg.pinv(B @ B.T + 1e-6 * np.eye(2)) @ exposure
            w = pd.Series(w.values - adj, index=ASSETS)
            # Preserve gross style after projection.
            long_g = max(0.0, w[w > 0].sum())
            short_g = max(0.0, -w[w < 0].sum())
            w = normalize_long_short(w, long_gross=min(CONFIG['LONG_GROSS'], long_g if long_g > 0 else CONFIG['LONG_GROSS']), short_gross=min(CONFIG['SHORT_GROSS'], short_g if short_g > 0 else CONFIG['SHORT_GROSS']))
        except Exception:
            break
    return w.reindex(ASSETS).fillna(0.0)


def apply_partial_adjustment(target_weights_by_day, alpha=None):
    alpha = CONFIG['PARTIAL_ADJUSTMENT_ALPHA'] if alpha is None else alpha
    out = {}
    prev = pd.Series(0.0, index=ASSETS)
    for dt in sorted(target_weights_by_day):
        target = pd.Series(target_weights_by_day[dt], index=ASSETS).fillna(0.0)
        actual = (1 - alpha) * prev + alpha * target
        out[pd.Timestamp(dt)] = actual
        prev = actual
    return out

print('Beta helpers ready.')

# %%
# ============================================================
# 8. Portfolio backtest engine
# ============================================================
def estimate_daily_cost_bps(w_new, w_old, dt):
    # Fixed taker/slippage cost plus a mild liquidity penalty for illiquid names.
    turnover_by_asset = (w_new - w_old).abs()
    base_cost = float(CONFIG['BASE_TRANSACTION_COST_BPS']) / 10000.0
    liq = liq_rank.loc[:pd.Timestamp(dt)].iloc[-1].reindex(ASSETS).fillna(len(ASSETS))
    liq_penalty_bps = (liq / max(1, len(ASSETS))).clip(0, 1) * 4.0
    cost = float((turnover_by_asset * (base_cost + liq_penalty_bps / 10000.0)).sum())
    return cost


def backtest_weight_function(weight_func, name, apply_partial=True, beta_neutral=False, apply_stops=False):
    dates = ret.index[ret.index >= pd.Timestamp(CONFIG['TRAIN_START'])]
    target_weights = {}
    diagnostics = []
    for dt in tqdm(dates[:-1], desc=f'weights {name}'):
        try:
            w = pd.Series(weight_func(dt), index=ASSETS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if beta_neutral:
                w = beta_neutralize(w, dt)
            w = normalize_long_short(w, long_gross=max(0.0, w[w > 0].sum()), short_gross=max(0.0, -w[w < 0].sum()))
            target_weights[pd.Timestamp(dt)] = w
        except Exception as exc:
            target_weights[pd.Timestamp(dt)] = pd.Series(0.0, index=ASSETS)
            diagnostics.append({'date': dt, 'error': repr(exc)})
    weights_by_day = apply_partial_adjustment(target_weights) if apply_partial else target_weights
    daily = pd.Series(0.0, index=dates, name=name)
    rows = []
    prev_w = pd.Series(0.0, index=ASSETS)
    eq = 1.0
    peak = 1.0
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i + 1]
        w = pd.Series(weights_by_day.get(pd.Timestamp(dt), pd.Series(0.0, index=ASSETS)), index=ASSETS).fillna(0.0)
        # Portfolio drawdown throttle.
        dd = eq / peak - 1.0
        if dd < float(CONFIG['PORTFOLIO_DD_CUT']):
            w = w * float(CONFIG['PORTFOLIO_DD_SCALE'])
        r = float(ret.loc[next_dt].reindex(ASSETS).fillna(0.0).dot(w))
        short_expo = float(-w[w < 0].sum())
        r -= short_expo * float(CONFIG['SHORT_FEE_ANNUAL']) / 365.0
        cost = estimate_daily_cost_bps(w, prev_w, dt)
        r -= cost
        daily.loc[next_dt] = r
        eq *= (1 + r)
        peak = max(peak, eq)
        rows.append({'date': next_dt, 'gross': float(w.abs().sum()), 'net': float(w.sum()), 'long_gross': float(w[w > 0].sum()), 'short_gross': short_expo, 'turnover': float((w - prev_w).abs().sum()), 'cost': cost, 'portfolio_dd': dd, 'n_long': int((w > 0).sum()), 'n_short': int((w < 0).sum())})
        prev_w = w
    log = pd.DataFrame(rows)
    if diagnostics:
        print(name, 'diagnostic errors:', diagnostics[:5])
    return daily.fillna(0.0).rename(name), log

print('Backtest engine ready.')

# %%
# ============================================================
# 9. Signal engines
# ============================================================
def common_signal_inputs(dt):
    dt = pd.Timestamp(dt)
    hist = ret.loc[:dt]
    px = close.loc[:dt]
    mom_7 = hist.tail(CONFIG['MOM_FAST']).sum()
    mom_21 = hist.tail(CONFIG['MOM_MED']).sum()
    mom_63 = hist.tail(CONFIG['MOM_SLOW']).sum()
    rev_1 = -hist.tail(CONFIG['REV_FAST']).sum()
    vol_21 = hist.tail(21).std() * np.sqrt(365)
    vol_63 = hist.tail(63).std() * np.sqrt(365)
    dd_21 = px.iloc[-1] / px.tail(21).max() - 1.0 if len(px) >= 21 else pd.Series(0.0, index=ASSETS)
    return mom_7, mom_21, mom_63, rev_1, vol_21, vol_63, dd_21


def strict_short_allowed(dt):
    dt = pd.Timestamp(dt)
    mom_7, mom_21, mom_63, rev_1, vol_21, vol_63, dd_21 = common_signal_inputs(dt)
    px = close.loc[:dt]
    below_20 = px.iloc[-1] < px.tail(20).mean() if len(px) >= 20 else pd.Series(False, index=ASSETS)
    below_50 = px.iloc[-1] < px.tail(50).mean() if len(px) >= 50 else pd.Series(False, index=ASSETS)
    rel_btc_21 = mom_21 - ret[BTC_ASSET].loc[:dt].tail(21).sum()
    funding_z = robust_z(funding.loc[:dt].tail(21).mean()) if funding.abs().sum().sum() > 0 else pd.Series(0.0, index=ASSETS)
    basis_z = robust_z(basis.loc[:dt].tail(21).mean()) if basis.notna().sum().sum() > 0 else pd.Series(0.0, index=ASSETS)
    crowded = (funding_z > 0.75) | (basis_z > 0.75)
    weak = (mom_21 < 0) & (mom_63 < 0) & (rel_btc_21 < 0) & below_20 & below_50
    if CONFIG['STRICT_SHORT_REQUIRE_FUNDING_OR_BASIS']:
        return (weak & crowded).reindex(ASSETS).fillna(False)
    return weak.reindex(ASSETS).fillna(False)


def w_funding_carry(dt):
    # Approximate spot-long/perp-short carry return uses funding minus basis change.
    if funding.abs().mean().mean() < float(CONFIG['FUNDING_MIN_ABS_DAILY']):
        return pd.Series(0.0, index=ASSETS)
    dt = pd.Timestamp(dt)
    f = funding.loc[:dt].tail(3).mean().reindex(ASSETS).fillna(0.0)
    b = basis.loc[:dt].tail(3).mean().reindex(ASSETS)
    oi_chg = open_interest.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).loc[:dt].tail(7).mean().reindex(ASSETS).fillna(0.0)
    score = robust_z(f) + 0.35 * robust_z(b.fillna(0.0)) + 0.20 * robust_z(oi_chg)
    eligible = score[(f > float(CONFIG['FUNDING_MIN_ABS_DAILY']))].nlargest(int(CONFIG['CARRY_TOP_N'])).index.tolist()
    if not eligible:
        return pd.Series(0.0, index=ASSETS)
    # Carry stream is market neutral internally, but we represent it as a synthetic return elsewhere.
    # Here use weights on synthetic carry asset returns, so weights are positive selected carry trades.
    return inverse_vol_weights(eligible, carry_asset_ret, dt, gross=1.0, lookback=30, cap=1.0 / max(1, int(CONFIG['CARRY_TOP_N'])))


def w_basis_carry(dt):
    if basis.notna().mean().mean() < 0.20:
        return pd.Series(0.0, index=ASSETS)
    dt = pd.Timestamp(dt)
    b = basis.loc[:dt].tail(3).mean().reindex(ASSETS)
    f = funding.loc[:dt].tail(3).mean().reindex(ASSETS).fillna(0.0)
    score = robust_z(b.fillna(0.0)) + 0.50 * robust_z(f)
    eligible = score[(b > float(CONFIG['BASIS_MIN_ABS']))].nlargest(int(CONFIG['CARRY_TOP_N'])).index.tolist()
    if not eligible:
        return pd.Series(0.0, index=ASSETS)
    return inverse_vol_weights(eligible, carry_asset_ret, dt, gross=1.0, lookback=30, cap=1.0 / max(1, int(CONFIG['CARRY_TOP_N'])))


def w_liquidity_aware_reversal(dt):
    dt = pd.Timestamp(dt)
    mom_7, mom_21, mom_63, rev_1, vol_21, vol_63, dd_21 = common_signal_inputs(dt)
    rank = liq_rank.loc[:dt].iloc[-1].reindex(ASSETS).fillna(len(ASSETS))
    liquid_core = rank <= int(CONFIG['LIQUID_CORE_N'])
    score_core = 0.50 * robust_z(mom_7) + 0.35 * robust_z(mom_21) - 0.15 * robust_z(vol_21)
    score_alt = 0.45 * robust_z(rev_1) + 0.35 * robust_z(mom_21) - 0.20 * robust_z(vol_21)
    score = score_alt.copy()
    score[liquid_core] = score_core[liquid_core]
    longs = score.nlargest(min(int(CONFIG['LONG_N']), len(score))).index.tolist()
    short_candidates = score.nsmallest(min(int(CONFIG['SHORT_N']) * 2, len(score))).index.tolist()
    allowed_short = strict_short_allowed(dt)
    shorts = [s for s in short_candidates if bool(allowed_short.get(s, False))][:int(CONFIG['SHORT_N'])]
    w = pd.Series(0.0, index=ASSETS)
    lw = inverse_vol_weights(longs, ret, dt, gross=float(CONFIG['LONG_GROSS']))
    sw = inverse_vol_weights(shorts, ret, dt, gross=float(CONFIG['SHORT_GROSS'])) if shorts else pd.Series(0.0, index=ASSETS)
    w = lw - sw.abs()
    return beta_neutralize(w, dt)


def w_stoploss_momentum(dt):
    dt = pd.Timestamp(dt)
    mom_7, mom_21, mom_63, rev_1, vol_21, vol_63, dd_21 = common_signal_inputs(dt)
    score = 0.35 * robust_z(mom_7) + 0.45 * robust_z(mom_21) + 0.20 * robust_z(mom_63) - 0.25 * robust_z(vol_21)
    longs = score.nlargest(min(int(CONFIG['LONG_N']), len(score))).index.tolist()
    allowed_short = strict_short_allowed(dt)
    shorts = [s for s in score.nsmallest(min(int(CONFIG['SHORT_N']) * 2, len(score))).index.tolist() if bool(allowed_short.get(s, False))][:int(CONFIG['SHORT_N'])]
    lw = inverse_vol_weights(longs, ret, dt, gross=float(CONFIG['LONG_GROSS']))
    sw = inverse_vol_weights(shorts, ret, dt, gross=float(CONFIG['SHORT_GROSS'])) if shorts else pd.Series(0.0, index=ASSETS)
    w = lw - sw.abs()
    # Stop overlays at position level.
    r5 = ret.loc[:dt].tail(5).sum().reindex(ASSETS).fillna(0.0)
    r10 = ret.loc[:dt].tail(10).sum().reindex(ASSETS).fillna(0.0)
    # Cut longs that recently violated adverse stop.
    w[(w > 0) & ((r5 < float(CONFIG['STOP_LOSS_5D'])) | (r10 < float(CONFIG['STOP_LOSS_10D'])))] *= 0.35
    # Cover shorts that recently ripped.
    w[(w < 0) & ((r5 > abs(float(CONFIG['STOP_LOSS_5D']))) | (r10 > abs(float(CONFIG['STOP_LOSS_10D']))))] *= 0.35
    return beta_neutralize(w, dt)


def w_beta_neutral_momentum(dt):
    w = w_stoploss_momentum(dt)
    return beta_neutralize(w, dt)

print('Signal engine functions ready.')

# %%
# ============================================================
# 10. Synthetic carry asset returns and carry streams
# ============================================================
# Synthetic carry return for long spot / short perp:
# approximately funding received minus change in perp/spot basis.
# This is intentionally disabled when real funding/basis data is absent.
carry_asset_ret = pd.DataFrame(0.0, index=close.index, columns=ASSETS)
carry_data_ok = (funding.abs().mean().mean() >= float(CONFIG['FUNDING_MIN_ABS_DAILY'])) and (basis.notna().mean().mean() >= 0.20)
if carry_data_ok:
    carry_asset_ret = (funding.reindex(close.index).fillna(0.0) - basis_change.reindex(close.index).fillna(0.0)).clip(-0.10, 0.10)
    print('Carry asset return enabled. funding abs mean=', funding.abs().mean().mean(), 'basis coverage=', basis.notna().mean().mean())
else:
    print('Carry asset return DISABLED: insufficient real funding/basis data.')


def backtest_carry_weight_function(weight_func, name):
    # Same engine as portfolio, but returns are synthetic carry_asset_ret rather than perp price ret.
    dates = carry_asset_ret.index[carry_asset_ret.index >= pd.Timestamp(CONFIG['TRAIN_START'])]
    target_weights = {}
    for dt in tqdm(dates[:-1], desc=f'weights {name}'):
        target_weights[pd.Timestamp(dt)] = pd.Series(weight_func(dt), index=ASSETS).fillna(0.0)
    weights_by_day = apply_partial_adjustment(target_weights)
    daily = pd.Series(0.0, index=dates, name=name)
    rows = []
    prev_w = pd.Series(0.0, index=ASSETS)
    for i, dt in enumerate(dates[:-1]):
        next_dt = dates[i+1]
        w = pd.Series(weights_by_day.get(pd.Timestamp(dt), pd.Series(0.0, index=ASSETS)), index=ASSETS).fillna(0.0)
        r = float(carry_asset_ret.loc[next_dt].reindex(ASSETS).fillna(0.0).dot(w))
        cost = estimate_daily_cost_bps(w, prev_w, dt)
        r -= cost
        daily.loc[next_dt] = r
        rows.append({'date': next_dt, 'gross': float(w.abs().sum()), 'turnover': float((w - prev_w).abs().sum()), 'cost': cost, 'n_carry': int((w != 0).sum())})
        prev_w = w
    return daily.fillna(0.0).rename(name), pd.DataFrame(rows)

if carry_data_ok:
    FUNDING_CARRY, FUNDING_CARRY_LOG = backtest_carry_weight_function(w_funding_carry, 'CRYPTO_FUNDING_CARRY')
    BASIS_CARRY, BASIS_CARRY_LOG = backtest_carry_weight_function(w_basis_carry, 'CRYPTO_BASIS_CARRY')
else:
    FUNDING_CARRY = pd.Series(0.0, index=ret.index, name='CRYPTO_FUNDING_CARRY')
    BASIS_CARRY = pd.Series(0.0, index=ret.index, name='CRYPTO_BASIS_CARRY')
    FUNDING_CARRY_LOG = pd.DataFrame()
    BASIS_CARRY_LOG = pd.DataFrame()

print('Carry streams built.')

# %%
# ============================================================
# 11. Pair stat-arb engine
# ============================================================
def estimate_half_life(spread):
    spread = pd.Series(spread).dropna()
    if len(spread) < 40:
        return np.nan
    y = spread.diff().dropna()
    x = spread.shift(1).loc[y.index]
    x = sm.add_constant(x)
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params.iloc[1]
        if beta >= 0:
            return np.nan
        return float(-np.log(2) / beta)
    except Exception:
        return np.nan


def select_pairs_for_date(dt, universe=None):
    dt = pd.Timestamp(dt)
    universe = universe or ASSETS[:min(len(ASSETS), int(CONFIG['MAX_ASSETS_FOR_PAIR_SEARCH']))]
    hist = np.log(close.loc[:dt, universe].tail(int(CONFIG['PAIR_LOOKBACK']))).replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')
    if hist.shape[0] < int(CONFIG['PAIR_LOOKBACK']) * 0.65 or hist.shape[1] < 4:
        return []
    cols = hist.columns.tolist()
    corr = hist.diff().corr()
    rows = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            c = float(corr.loc[a, b]) if pd.notna(corr.loc[a, b]) else 0.0
            if abs(c) < float(CONFIG['PAIR_MIN_CORR']):
                continue
            try:
                pval = coint(hist[a], hist[b])[1]
            except Exception:
                continue
            if pval > float(CONFIG['PAIR_MAX_COINTEGRATION_P']):
                continue
            X = sm.add_constant(hist[b])
            try:
                hedge = sm.OLS(hist[a], X).fit().params.iloc[1]
            except Exception:
                continue
            spread = hist[a] - hedge * hist[b]
            hl = estimate_half_life(spread)
            if pd.isna(hl) or hl < float(CONFIG['PAIR_MIN_HALFLIFE']) or hl > float(CONFIG['PAIR_MAX_HALFLIFE']):
                continue
            z = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0.0
            rows.append({'a': a, 'b': b, 'corr': c, 'pval': pval, 'hedge': hedge, 'half_life': hl, 'z': z, 'score': abs(z) * abs(c) / max(pval, 1e-4)})
    rows = sorted(rows, key=lambda x: x['score'], reverse=True)
    return rows[:int(CONFIG['MAX_ACTIVE_PAIRS'])]


def w_pair_statarb(dt):
    pairs = select_pairs_for_date(dt)
    w = pd.Series(0.0, index=ASSETS)
    active = 0
    for p in pairs:
        z = p['z']
        if abs(z) < float(CONFIG['PAIR_Z_ENTRY']) or abs(z) > float(CONFIG['PAIR_Z_STOP']):
            continue
        a, b, h = p['a'], p['b'], float(p['hedge'])
        # spread = log(a) - h log(b). If z high, a rich vs b: short a, long b.
        pair_gross = 1.0 / max(1, int(CONFIG['MAX_ACTIVE_PAIRS']))
        if z > 0:
            w[a] -= pair_gross * 0.5
            w[b] += pair_gross * 0.5
        else:
            w[a] += pair_gross * 0.5
            w[b] -= pair_gross * 0.5
        active += 1
    if active == 0:
        return w
    return normalize_long_short(w, long_gross=0.50, short_gross=0.50, max_total_gross=1.0)

PAIR_STATARB, PAIR_STATARB_LOG = backtest_weight_function(w_pair_statarb, 'CRYPTO_PAIR_STATARB_COINTEGRATED', apply_partial=True, beta_neutral=False)
print('Pair stat-arb built.')

# %%
# ============================================================
# 12. Directional / relative-value engines
# ============================================================
LIQUIDITY_REVERSAL, LIQUIDITY_REVERSAL_LOG = backtest_weight_function(w_liquidity_aware_reversal, 'CRYPTO_LIQUIDITY_AWARE_REVERSAL_BETA_NEUTRAL', apply_partial=True, beta_neutral=True)
STOPLOSS_MOMENTUM, STOPLOSS_MOMENTUM_LOG = backtest_weight_function(w_stoploss_momentum, 'CRYPTO_STOPLOSS_MOMENTUM_BETA_NEUTRAL', apply_partial=True, beta_neutral=True)
BETA_NEUTRAL_MOMENTUM, BETA_NEUTRAL_MOMENTUM_LOG = backtest_weight_function(w_beta_neutral_momentum, 'CRYPTO_BETA_NEUTRAL_MOMENTUM', apply_partial=True, beta_neutral=True)

# Risk-managed variants.
STOPLOSS_MOMENTUM_VOL = apply_vol_target(STOPLOSS_MOMENTUM, CONFIG['TARGET_VOL'], 30, CONFIG['MAX_LEVERAGE']).rename('CRYPTO_STOPLOSS_MOMENTUM_VOL_TARGET')
STOPLOSS_MOMENTUM_DVOL = apply_downside_vol_target(STOPLOSS_MOMENTUM, CONFIG['DOWNSIDE_TARGET_VOL'], 30, CONFIG['MAX_LEVERAGE']).rename('CRYPTO_STOPLOSS_MOMENTUM_DOWNSIDE_VOL')
LIQUIDITY_REVERSAL_VOL = apply_vol_target(LIQUIDITY_REVERSAL, CONFIG['TARGET_VOL'], 30, CONFIG['MAX_LEVERAGE']).rename('CRYPTO_LIQUIDITY_REVERSAL_VOL_TARGET')

print('Directional/relative-value engines built.')

# %%
# ============================================================
# 13. Combine streams, evaluate, fold diagnostics
# ============================================================
stream_df = pd.concat({
    'CRYPTO_FUNDING_CARRY': FUNDING_CARRY,
    'CRYPTO_BASIS_CARRY': BASIS_CARRY,
    'CRYPTO_PAIR_STATARB_COINTEGRATED': PAIR_STATARB,
    'CRYPTO_LIQUIDITY_AWARE_REVERSAL_BETA_NEUTRAL': LIQUIDITY_REVERSAL,
    'CRYPTO_LIQUIDITY_REVERSAL_VOL_TARGET': LIQUIDITY_REVERSAL_VOL,
    'CRYPTO_STOPLOSS_MOMENTUM_BETA_NEUTRAL': STOPLOSS_MOMENTUM,
    'CRYPTO_STOPLOSS_MOMENTUM_VOL_TARGET': STOPLOSS_MOMENTUM_VOL,
    'CRYPTO_STOPLOSS_MOMENTUM_DOWNSIDE_VOL': STOPLOSS_MOMENTUM_DVOL,
    'CRYPTO_BETA_NEUTRAL_MOMENTUM': BETA_NEUTRAL_MOMENTUM,
}, axis=1).sort_index().fillna(0.0)

# Remove completely flat streams from selectors, but keep in report.
active_stream_cols = [c for c in stream_df.columns if stream_df[c].abs().sum() > 1e-9]

periods = [
    ('Train', CONFIG['TRAIN_START'], CONFIG['TRAIN_END']),
    ('Validation', CONFIG['VALIDATION_START'], CONFIG['VALIDATION_END']),
    ('Test', CONFIG['TEST_START'], CONFIG['TEST_END']),
    ('Full', CONFIG['TRAIN_START'], CONFIG['TEST_END']),
]
perf_rows = []
for name in stream_df.columns:
    for period, start, end in periods:
        row = {'strategy': name, 'period': period, **perf_stats(stream_df[name], start, end)}
        perf_rows.append(row)
STREAM_PERF = pd.DataFrame(perf_rows)
print('Stream performance:')
display(STREAM_PERF.sort_values(['period','sharpe'], ascending=[True, False]))

# Rolling fold diagnostics over train+validation, then final test untouched.
def fold_ranges():
    out = []
    start = pd.Timestamp(CONFIG['FOLD_START'])
    end_lim = pd.Timestamp(CONFIG['VALIDATION_END'])
    fold_days = int(CONFIG['FOLD_DAYS'])
    step = int(CONFIG['FOLD_STEP_DAYS'])
    i = 0
    while start + pd.Timedelta(days=fold_days) <= end_lim:
        end = start + pd.Timedelta(days=fold_days)
        out.append((i, start, end))
        start += pd.Timedelta(days=step)
        i += 1
    return out

fold_rows = []
for fid, fs, fe in fold_ranges():
    for name in stream_df.columns:
        fold_rows.append({'fold_id': fid, 'fold_start': fs, 'fold_end': fe, 'strategy': name, **perf_stats(stream_df[name], fs, fe)})
FOLD_PERF = pd.DataFrame(fold_rows)
if not FOLD_PERF.empty:
    FOLD_SUMMARY = FOLD_PERF.groupby('strategy').agg(
        fold_count=('fold_id','nunique'),
        median_fold_sharpe=('sharpe','median'),
        mean_fold_sharpe=('sharpe','mean'),
        min_fold_sharpe=('sharpe','min'),
        pass_rate_positive=('sharpe', lambda x: float(np.mean(pd.Series(x) > 0))),
        pass_rate_1=('sharpe', lambda x: float(np.mean(pd.Series(x) > 1.0))),
        median_return=('ann_return','median'),
        median_dd=('max_dd','median'),
    ).reset_index()
    FOLD_SUMMARY['fold_score'] = (
        1.5 * FOLD_SUMMARY['median_fold_sharpe'].fillna(-9) +
        1.0 * FOLD_SUMMARY['pass_rate_positive'].fillna(0) +
        1.0 * FOLD_SUMMARY['pass_rate_1'].fillna(0) +
        0.5 * FOLD_SUMMARY['median_return'].fillna(-9) -
        1.0 * FOLD_SUMMARY['median_dd'].abs().fillna(0)
    )
else:
    FOLD_SUMMARY = pd.DataFrame()
print('Fold summary:')
display(FOLD_SUMMARY.sort_values('fold_score', ascending=False) if not FOLD_SUMMARY.empty else FOLD_SUMMARY)

# %%
# ============================================================
# 14. V26 risk-on-only final strategy builder
# ============================================================
# Core idea from V25 diagnostics:
#   - Relative-value / beta-neutral crypto engines have >2.5 Sharpe in risk-on slices.
#   - Neutral and risk-off regimes dilute or damage performance.
# V26 therefore trades only when a causal, prior-day risk-on gate is true and stays flat otherwise.

V26_PERIODS = [
    ('Train', CONFIG['TRAIN_START'], CONFIG['TRAIN_END']),
    ('Validation', CONFIG['VALIDATION_START'], CONFIG['VALIDATION_END']),
    ('Test', CONFIG['TEST_START'], CONFIG['TEST_END']),
    ('Full', CONFIG['TRAIN_START'], CONFIG['TEST_END']),
]

# Candidate streams that showed the best V25 behavior. Only keep columns that exist and are non-flat.
V26_BASE_CANDIDATES = [c for c in [
    'CRYPTO_LIQUIDITY_AWARE_REVERSAL_BETA_NEUTRAL',
    'CRYPTO_LIQUIDITY_REVERSAL_VOL_TARGET',
    'CRYPTO_STOPLOSS_MOMENTUM_BETA_NEUTRAL',
    'CRYPTO_STOPLOSS_MOMENTUM_VOL_TARGET',
    'CRYPTO_STOPLOSS_MOMENTUM_DOWNSIDE_VOL',
    'CRYPTO_BETA_NEUTRAL_MOMENTUM',
] if c in stream_df.columns and stream_df[c].abs().sum() > 1e-9]

# Add conservative equal/inverse-vol blends from candidates that passed fold/validation filters in V25.
base_selector = STREAM_PERF[STREAM_PERF['period'].eq('Validation')].copy()
if 'FOLD_SUMMARY' in globals() and not FOLD_SUMMARY.empty:
    base_selector = base_selector.merge(
        FOLD_SUMMARY[['strategy','fold_score','median_fold_sharpe','pass_rate_positive','pass_rate_1']],
        on='strategy', how='left'
    )
else:
    base_selector['median_fold_sharpe'] = base_selector['sharpe']
    base_selector['pass_rate_positive'] = np.nan
    base_selector['pass_rate_1'] = np.nan

base_selector['eligible_v26_base'] = (
    base_selector['strategy'].isin(V26_BASE_CANDIDATES)
    & (base_selector['sharpe'].fillna(-9) >= float(CONFIG['MIN_VALIDATION_SHARPE']))
    & (base_selector['median_fold_sharpe'].fillna(-9) >= float(CONFIG.get('V26_MIN_MEDIAN_FOLD_SHARPE', 0.25)))
)
base_selector['base_score'] = (
    1.25 * base_selector['sharpe'].fillna(-9)
    + 1.25 * base_selector['median_fold_sharpe'].fillna(-9)
    + 0.75 * base_selector['pass_rate_positive'].fillna(0.0)
    + 0.50 * base_selector['pass_rate_1'].fillna(0.0)
    + 0.20 * base_selector['ann_return'].fillna(-9)
    - 0.75 * base_selector['max_dd'].abs().fillna(0.0)
)
base_selector = base_selector.sort_values('base_score', ascending=False).reset_index(drop=True)
V26_BASE_SURVIVORS = base_selector[base_selector['eligible_v26_base']]['strategy'].head(4).tolist()

if len(V26_BASE_SURVIVORS) >= 2:
    V26_EQUAL_SURVIVOR_BLEND = stream_df[V26_BASE_SURVIVORS].mean(axis=1).rename('V26_EQUAL_SURVIVOR_BLEND')
    vols = stream_df[V26_BASE_SURVIVORS].loc[CONFIG['VALIDATION_START']:CONFIG['VALIDATION_END']].std().replace(0, np.nan)
    inv = (1.0 / vols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    V26_IVOL_WEIGHTS = inv / inv.sum() if inv.sum() > 0 else pd.Series(1.0 / len(V26_BASE_SURVIVORS), index=V26_BASE_SURVIVORS)
    V26_INVOL_SURVIVOR_BLEND = stream_df[V26_BASE_SURVIVORS].dot(V26_IVOL_WEIGHTS).rename('V26_INVOL_SURVIVOR_BLEND')
else:
    V26_EQUAL_SURVIVOR_BLEND = pd.Series(0.0, index=stream_df.index, name='V26_EQUAL_SURVIVOR_BLEND')
    V26_INVOL_SURVIVOR_BLEND = V26_EQUAL_SURVIVOR_BLEND.rename('V26_INVOL_SURVIVOR_BLEND')
    V26_IVOL_WEIGHTS = pd.Series(dtype=float)

v26_base_streams = pd.concat([stream_df[V26_BASE_CANDIDATES], V26_EQUAL_SURVIVOR_BLEND, V26_INVOL_SURVIVOR_BLEND], axis=1).fillna(0.0)
V26_ALL_BASE_STREAMS = list(v26_base_streams.columns)

print('V26 base candidate streams:', V26_BASE_CANDIDATES)
print('V26 base survivors:', V26_BASE_SURVIVORS)
if len(V26_IVOL_WEIGHTS):
    print('V26 inverse-vol survivor weights:')
    display(V26_IVOL_WEIGHTS.to_frame('weight'))
display(base_selector.head(20))


def rolling_sharpe_daily(x, window=21, periods=365):
    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mu = x.rolling(window, min_periods=max(8, window // 2)).mean()
    sd = x.rolling(window, min_periods=max(8, window // 2)).std().replace(0, np.nan)
    return (mu / sd * np.sqrt(periods)).replace([np.inf, -np.inf], np.nan)


def consecutive_true(mask, days):
    mask = pd.Series(mask).astype(bool)
    if int(days) <= 0:
        return mask
    return mask.rolling(int(days), min_periods=int(days)).sum().fillna(0) >= int(days)


def v26_causal_risk_on_mask(base_stream=None, risk_on_score_min=0.70, breadth50_min=0.50,
                            confirmation_days=0, trail_window=0, trail_sharpe_min=0.0):
    """Build a no-lookahead active mask for returns indexed at t.

    Returns at t come from positions decided before t, so all gate inputs are shifted by one day.
    """
    r = regime.copy().reindex(stream_df.index).ffill()
    gate_today = (
        (r['risk_on_score'] >= float(risk_on_score_min))
        & (r['breadth_50'] >= float(breadth50_min))
        & (r['btc_above_50'] > 0.5)
        & (r['btc_above_200'] > 0.5)
        & (r['eth_above_50'] > 0.5)
        & (r['btc_ret_21'] > 0.0)
        & (r['btc_dd_63'] > -0.20)
    )
    if int(confirmation_days) > 0:
        gate_today = consecutive_true(gate_today, int(confirmation_days))
    active = gate_today.shift(1).fillna(False)

    if base_stream is not None and int(trail_window) > 0:
        sh = rolling_sharpe_daily(base_stream, int(trail_window)).shift(1).reindex(stream_df.index)
        active = active & (sh >= float(trail_sharpe_min)).fillna(False)
    return active.astype(bool).reindex(stream_df.index).fillna(False)


def apply_v26_gate(base_ret, mask, name):
    base_ret = series(base_ret).reindex(stream_df.index).fillna(0.0)
    mask = pd.Series(mask).reindex(base_ret.index).fillna(False).astype(bool)
    return (base_ret * mask.astype(float)).rename(name)


def v26_active_stats(ret, mask, start=None, end=None):
    s = series(ret)
    m = pd.Series(mask).reindex(s.index).fillna(False).astype(bool)
    if start is not None:
        s = s[s.index >= pd.Timestamp(start)]
        m = m[m.index >= pd.Timestamp(start)]
    if end is not None:
        s = s[s.index <= pd.Timestamp(end)]
        m = m[m.index <= pd.Timestamp(end)]
    active_rate = float(m.mean()) if len(m) else np.nan
    active_days = int(m.sum()) if len(m) else 0
    st_calendar = perf_stats(s, None, None)
    st_active = perf_stats(s[m], None, None) if active_days >= 20 else {k: np.nan for k in st_calendar.keys()}
    return {
        'calendar_sharpe': st_calendar['sharpe'],
        'calendar_ann_return': st_calendar['ann_return'],
        'calendar_ann_vol': st_calendar['ann_vol'],
        'calendar_max_dd': st_calendar['max_dd'],
        'active_sharpe': st_active['sharpe'],
        'active_ann_return': st_active['ann_return'],
        'active_ann_vol': st_active['ann_vol'],
        'active_max_dd': st_active['max_dd'],
        'active_rate': active_rate,
        'active_days': active_days,
        'n_days': int(len(s)),
    }


def apply_v26_vol_target(ret, target_vol=None, name=None):
    if target_vol is None or pd.isna(target_vol):
        out = series(ret).copy()
        out.name = name or getattr(ret, 'name', 'ret')
        return out
    return apply_vol_target(series(ret), float(target_vol), lookback=30, max_leverage=float(CONFIG.get('V26_MAX_LEVERAGE', 1.5))).rename(name or getattr(ret, 'name', 'ret'))

# Generate a controlled grid. The selector will use only train/validation/folds.
v26_candidate_returns = OrderedDict()
v26_candidate_masks = OrderedDict()
v26_candidate_meta = []

risk_scores = list(CONFIG.get('V26_RISK_ON_SCORE_GRID', [0.65, 0.70, 0.75]))
breadths = list(CONFIG.get('V26_BREADTH_50_GRID', [0.45, 0.50, 0.55]))
confirm_days = list(CONFIG.get('V26_CONFIRMATION_DAYS_GRID', [0, 2, 5]))
trail_windows = list(CONFIG.get('V26_TRAIL_WINDOWS', [0, 21, 63]))
trail_mins = list(CONFIG.get('V26_TRAIL_SHARPE_MIN_GRID', [-0.25, 0.0, 0.25]))
target_vols = list(CONFIG.get('V26_TARGET_VOL_GRID', [None, 0.08, 0.12, 0.16]))

# Keep the grid bounded: pair each trail window with sensible trail thresholds.
for stream_name in V26_ALL_BASE_STREAMS:
    base = v26_base_streams[stream_name]
    for score_thr in risk_scores:
        for breadth_thr in breadths:
            for cd in confirm_days:
                for tw in trail_windows:
                    for tmin in ([0.0] if int(tw) == 0 else trail_mins):
                        mask = v26_causal_risk_on_mask(
                            base_stream=base,
                            risk_on_score_min=score_thr,
                            breadth50_min=breadth_thr,
                            confirmation_days=cd,
                            trail_window=tw,
                            trail_sharpe_min=tmin,
                        )
                        raw_name = f"V26_{stream_name}_riskon_s{int(score_thr*100)}_b{int(breadth_thr*100)}_c{int(cd)}_tw{int(tw)}_tm{int((tmin+1)*100)}"
                        raw_ret = apply_v26_gate(base, mask, raw_name)
                        for tv in target_vols:
                            name = raw_name if tv is None else f"{raw_name}_vol{int(float(tv)*100):02d}"
                            vt_ret = apply_v26_vol_target(raw_ret, tv, name=name)
                            v26_candidate_returns[name] = vt_ret
                            v26_candidate_masks[name] = mask
                            v26_candidate_meta.append({
                                'strategy': name,
                                'base_stream': stream_name,
                                'risk_on_score_min': score_thr,
                                'breadth50_min': breadth_thr,
                                'confirmation_days': cd,
                                'trail_window': tw,
                                'trail_sharpe_min': tmin,
                                'target_vol': tv,
                            })

v26_returns_df = pd.concat(v26_candidate_returns, axis=1).fillna(0.0)
V26_CANDIDATE_META = pd.DataFrame(v26_candidate_meta)
print('V26 candidate variants:', v26_returns_df.shape[1])
display(V26_CANDIDATE_META.head())

# %%
# ============================================================
# 15. V26 fold-stable selector and final test report
# ============================================================
# No test information is used for selection. Test is reported only after the V26 champion is frozen.

v26_perf_rows = []
for name in v26_returns_df.columns:
    ret_s = v26_returns_df[name]
    mask_s = v26_candidate_masks[name]
    for period, start, end in V26_PERIODS:
        st = perf_stats(ret_s, start, end)
        ast = v26_active_stats(ret_s, mask_s, start, end)
        v26_perf_rows.append({'strategy': name, 'period': period, **st, **{f'v26_{k}': v for k, v in ast.items()}})
V26_PERF = pd.DataFrame(v26_perf_rows)

# Fold diagnostics over train + validation.
v26_fold_rows = []
for name in v26_returns_df.columns:
    ret_s = v26_returns_df[name]
    mask_s = v26_candidate_masks[name]
    for fid, fs, fe in fold_ranges():
        st = perf_stats(ret_s, fs, fe)
        ast = v26_active_stats(ret_s, mask_s, fs, fe)
        v26_fold_rows.append({'fold_id': fid, 'fold_start': fs, 'fold_end': fe, 'strategy': name, **st, **{f'v26_{k}': v for k, v in ast.items()}})
V26_FOLD_PERF = pd.DataFrame(v26_fold_rows)

if not V26_FOLD_PERF.empty:
    V26_FOLD_SUMMARY = (
        V26_FOLD_PERF.groupby('strategy')
        .agg(
            fold_count=('fold_id', 'nunique'),
            median_fold_sharpe=('sharpe', 'median'),
            mean_fold_sharpe=('sharpe', 'mean'),
            std_fold_sharpe=('sharpe', 'std'),
            min_fold_sharpe=('sharpe', 'min'),
            pass_rate_positive=('sharpe', lambda x: float(np.mean(pd.Series(x) > 0))),
            pass_rate_1=('sharpe', lambda x: float(np.mean(pd.Series(x) > 1.0))),
            pass_rate_2=('sharpe', lambda x: float(np.mean(pd.Series(x) > 2.0))),
            median_fold_return=('ann_return', 'median'),
            median_fold_dd=('max_dd', 'median'),
            median_active_rate=('v26_active_rate', 'median'),
            median_active_days=('v26_active_days', 'median'),
        )
        .reset_index()
    )
else:
    V26_FOLD_SUMMARY = pd.DataFrame()

v26_val = V26_PERF[V26_PERF['period'].eq('Validation')].copy()
V26_SELECTOR = v26_val.merge(V26_FOLD_SUMMARY, on='strategy', how='left') if not V26_FOLD_SUMMARY.empty else v26_val.copy()
V26_SELECTOR = V26_SELECTOR.merge(V26_CANDIDATE_META, on='strategy', how='left')

# Production-grade eligibility: active enough, positive validation, stable folds, and controlled drawdown.
V26_SELECTOR['eligible'] = (
    (V26_SELECTOR['sharpe'].fillna(-9) > 0)
    & (V26_SELECTOR['ann_return'].fillna(-9) > 0)
    & (V26_SELECTOR['v26_active_days'].fillna(0) >= int(CONFIG.get('V26_MIN_ACTIVE_DAYS_VALIDATION', 20)))
    & (V26_SELECTOR['v26_active_rate'].fillna(0) >= float(CONFIG.get('V26_MIN_ACTIVE_RATE_VALIDATION', 0.05)))
    & (V26_SELECTOR.get('median_fold_sharpe', pd.Series(0, index=V26_SELECTOR.index)).fillna(-9) >= float(CONFIG.get('V26_MIN_MEDIAN_FOLD_SHARPE', 0.25)))
    & (V26_SELECTOR.get('pass_rate_positive', pd.Series(0, index=V26_SELECTOR.index)).fillna(0) >= 0.50)
)

# Score balances validation, folds, active rate, return, and drawdown. Complexity penalty discourages over-gated/noisy variants.
V26_SELECTOR['complexity_penalty'] = (
    0.05 * (V26_SELECTOR['confirmation_days'].fillna(0).astype(float) > 0).astype(float)
    + 0.05 * (V26_SELECTOR['trail_window'].fillna(0).astype(float) > 0).astype(float)
    + 0.03 * (V26_SELECTOR['target_vol'].notna()).astype(float)
)
V26_SELECTOR['selector_score'] = (
    1.50 * V26_SELECTOR['sharpe'].fillna(-9)
    + 1.75 * V26_SELECTOR.get('median_fold_sharpe', pd.Series(0, index=V26_SELECTOR.index)).fillna(-9)
    + 1.00 * V26_SELECTOR.get('pass_rate_positive', pd.Series(0, index=V26_SELECTOR.index)).fillna(0)
    + 0.75 * V26_SELECTOR.get('pass_rate_1', pd.Series(0, index=V26_SELECTOR.index)).fillna(0)
    + 0.25 * V26_SELECTOR.get('pass_rate_2', pd.Series(0, index=V26_SELECTOR.index)).fillna(0)
    + 0.35 * V26_SELECTOR['ann_return'].fillna(-9)
    - 1.25 * V26_SELECTOR['max_dd'].abs().fillna(0)
    + 0.25 * V26_SELECTOR['v26_active_rate'].fillna(0)
    - V26_SELECTOR['complexity_penalty']
)
V26_SELECTOR = V26_SELECTOR.sort_values(['eligible','selector_score'], ascending=[False, False]).reset_index(drop=True)

if not V26_SELECTOR['eligible'].any():
    print('WARNING: no V26 variant met full production-grade eligibility. Selecting best available variant for reporting only.')

V26_CHAMPION = V26_SELECTOR.iloc[0]['strategy'] if len(V26_SELECTOR) else v26_returns_df.columns[0]
V26_CHAMPION_RETURN = v26_returns_df[V26_CHAMPION].rename('V26_RISK_ON_ONLY_CHAMPION')
V26_CHAMPION_MASK = v26_candidate_masks[V26_CHAMPION]

# Comparison versus base candidate streams, flat, and survivor blends.
v26_compare_refs = list(dict.fromkeys(V26_BASE_CANDIDATES + ['V26_EQUAL_SURVIVOR_BLEND','V26_INVOL_SURVIVOR_BLEND']))
v26_reference_df = v26_base_streams.copy()
v26_reference_df['CASH_FLAT'] = 0.0
v26_reference_df['V26_RISK_ON_ONLY_CHAMPION'] = V26_CHAMPION_RETURN

V26_FINAL_PERF_ROWS = []
for name in ['V26_RISK_ON_ONLY_CHAMPION'] + [c for c in v26_reference_df.columns if c != 'V26_RISK_ON_ONLY_CHAMPION']:
    s = v26_reference_df[name]
    for period, start, end in V26_PERIODS:
        V26_FINAL_PERF_ROWS.append({'strategy': name, 'period': period, **perf_stats(s, start, end)})
V26_FINAL_PERF = pd.DataFrame(V26_FINAL_PERF_ROWS)

compare_rows = []
for ref in [c for c in v26_reference_df.columns if c != 'V26_RISK_ON_ONLY_CHAMPION']:
    for period, start, end in [('Validation', CONFIG['VALIDATION_START'], CONFIG['VALIDATION_END']), ('Test', CONFIG['TEST_START'], CONFIG['TEST_END'])]:
        c = perf_stats(V26_CHAMPION_RETURN, start, end)
        r = perf_stats(v26_reference_df[ref], start, end)
        active = pd.concat([V26_CHAMPION_RETURN.rename('champ'), v26_reference_df[ref].rename('ref')], axis=1).loc[pd.Timestamp(start):]
        if end is not None:
            active = active.loc[:pd.Timestamp(end)]
        diff = active['champ'] - active['ref']
        compare_rows.append({
            'champion': 'V26_RISK_ON_ONLY_CHAMPION',
            'reference': ref,
            'period': period,
            'champion_sharpe': c['sharpe'],
            'reference_sharpe': r['sharpe'],
            'sharpe_diff': c['sharpe'] - r['sharpe'] if pd.notna(c['sharpe']) and pd.notna(r['sharpe']) else np.nan,
            'champion_ann_return': c['ann_return'],
            'reference_ann_return': r['ann_return'],
            'return_diff': c['ann_return'] - r['ann_return'] if pd.notna(c['ann_return']) and pd.notna(r['ann_return']) else np.nan,
            'champion_max_dd': c['max_dd'],
            'reference_max_dd': r['max_dd'],
            'information_ratio_vs_ref': ann_sharpe(diff),
        })
V26_CHAMPION_COMPARE = pd.DataFrame(compare_rows)

V26_ACTIVE_SUMMARY_ROWS = []
for period, start, end in V26_PERIODS:
    V26_ACTIVE_SUMMARY_ROWS.append({'period': period, **v26_active_stats(V26_CHAMPION_RETURN, V26_CHAMPION_MASK, start, end)})
V26_ACTIVE_SUMMARY = pd.DataFrame(V26_ACTIVE_SUMMARY_ROWS)

# Regime attribution for champion and top base streams.
def v26_regime_attribution(ret_series, name):
    s = series(ret_series)
    df = pd.DataFrame({'ret': s}).join(regime['regime'], how='left').dropna()
    rows = []
    for reg, g in df.groupby('regime'):
        rows.append({'strategy': name, 'regime': reg, **perf_stats(g['ret'], None, None)})
    return pd.DataFrame(rows)

regime_rows = [v26_regime_attribution(V26_CHAMPION_RETURN, 'V26_RISK_ON_ONLY_CHAMPION')]
for ref in v26_compare_refs[:6]:
    if ref in v26_reference_df.columns:
        regime_rows.append(v26_regime_attribution(v26_reference_df[ref], ref))
V26_REGIME_ATTRIB = pd.concat(regime_rows, ignore_index=True) if regime_rows else pd.DataFrame()

# Latest production-style signal snapshot.
latest_dt = v26_reference_df.index.max()
latest_state = regime.loc[:latest_dt].iloc[-1].to_dict()
latest_signal = pd.DataFrame([{
    'asof_date': latest_dt,
    'champion': V26_CHAMPION,
    'active_today_for_next_bar': bool(V26_CHAMPION_MASK.reindex([latest_dt]).fillna(False).iloc[0]),
    'risk_on_score': latest_state.get('risk_on_score', np.nan),
    'regime': latest_state.get('regime', 'unknown'),
    'btc_above_50': latest_state.get('btc_above_50', np.nan),
    'btc_above_200': latest_state.get('btc_above_200', np.nan),
    'eth_above_50': latest_state.get('eth_above_50', np.nan),
    'breadth_50': latest_state.get('breadth_50', np.nan),
    'btc_ret_21': latest_state.get('btc_ret_21', np.nan),
    'btc_dd_63': latest_state.get('btc_dd_63', np.nan),
}])

print('V26 champion:', V26_CHAMPION)
print('Top V26 selector rows:')
display(V26_SELECTOR.head(30))
print('V26 final performance:')
display(V26_FINAL_PERF.sort_values(['period','sharpe'], ascending=[True, False]))
print('V26 active-day summary:')
display(V26_ACTIVE_SUMMARY)
print('V26 champion comparison:')
display(V26_CHAMPION_COMPARE)
print('V26 regime attribution:')
display(V26_REGIME_ATTRIB.sort_values(['strategy','sharpe'], ascending=[True, False]) if not V26_REGIME_ATTRIB.empty else V26_REGIME_ATTRIB)
print('Latest production signal snapshot:')
display(latest_signal)

plot_items = {'V26_RISK_ON_ONLY_CHAMPION': V26_CHAMPION_RETURN}
for ref in v26_compare_refs[:5]:
    if ref in v26_reference_df.columns:
        plot_items[ref] = v26_reference_df[ref]
plot_equity_curve(plot_items, 'V26 risk-on-only champion vs base streams — Test', start=CONFIG['TEST_START'], end=CONFIG['TEST_END'])

# %%
# ============================================================
# 16. V26 production-grade exports
# ============================================================
# These files are intentionally audit-friendly. The notebook remains a research notebook,
# but it exports the frozen champion, mask, daily returns, selector, and latest signal for paper-trading conversion.

v26_daily_returns = pd.DataFrame({
    'V26_RISK_ON_ONLY_CHAMPION': V26_CHAMPION_RETURN,
    'V26_ACTIVE_MASK': V26_CHAMPION_MASK.astype(int),
}).join(v26_reference_df[[c for c in v26_reference_df.columns if c != 'V26_RISK_ON_ONLY_CHAMPION']], how='left')

metadata = {
    'name': CONFIG['NAME'],
    'version': 'V26_RISK_ON_ONLY_CRYPTO_RELVAL_PRODUCTION',
    'exchange_id': EXCHANGE_ID,
    'champion': V26_CHAMPION,
    'base_candidates': V26_BASE_CANDIDATES,
    'base_survivors': V26_BASE_SURVIVORS,
    'assets': ASSETS,
    'test_start': CONFIG['TEST_START'],
    'test_end': CONFIG['TEST_END'],
    'selection_rule': 'train_validation_fold_selector_only__test_report_only',
    'latest_signal': latest_signal.to_dict('records'),
    'config': CONFIG,
}

metadata_path = OUTDIR / 'v26_risk_on_only_crypto_relval_metadata.json'
metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

returns_csv = OUTDIR / 'v26_risk_on_only_crypto_relval_daily_returns.csv'
v26_daily_returns.to_csv(returns_csv)

latest_signal_csv = OUTDIR / 'v26_latest_signal.csv'
latest_signal.to_csv(latest_signal_csv, index=False)

xlsx_path = OUTDIR / 'v26_risk_on_only_crypto_relval_results.xlsx'
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    pd.DataFrame([CONFIG]).T.reset_index().rename(columns={'index':'parameter', 0:'value'}).to_excel(writer, 'Config', index=False)
    UNIVERSE.to_excel(writer, 'Universe Raw', index=False)
    FETCH_REPORT.to_excel(writer, 'Fetch Report', index=False)
    STREAM_PERF.to_excel(writer, 'V25 Stream Perf', index=False)
    FOLD_PERF.to_excel(writer, 'V25 Fold Perf', index=False)
    FOLD_SUMMARY.to_excel(writer, 'V25 Fold Summary', index=False)
    base_selector.to_excel(writer, 'V26 Base Selector', index=False)
    pd.DataFrame({'stream': V26_BASE_SURVIVORS}).to_excel(writer, 'V26 Base Survivors', index=False)
    V26_CANDIDATE_META.to_excel(writer, 'V26 Candidate Meta', index=False)
    V26_PERF.to_excel(writer, 'V26 Candidate Perf', index=False)
    V26_FOLD_PERF.to_excel(writer, 'V26 Fold Perf', index=False)
    V26_FOLD_SUMMARY.to_excel(writer, 'V26 Fold Summary', index=False)
    V26_SELECTOR.to_excel(writer, 'V26 Selector', index=False)
    V26_FINAL_PERF.to_excel(writer, 'V26 Final Perf', index=False)
    V26_ACTIVE_SUMMARY.to_excel(writer, 'V26 Active Summary', index=False)
    V26_CHAMPION_COMPARE.to_excel(writer, 'V26 Champion Compare', index=False)
    V26_REGIME_ATTRIB.to_excel(writer, 'V26 Regime Attribution', index=False)
    latest_signal.to_excel(writer, 'Latest Signal', index=False)
    v26_daily_returns.reset_index(names='date').to_excel(writer, 'Daily Returns', index=False)
    pd.DataFrame({'asset': ASSETS, 'base': BASES}).to_excel(writer, 'Assets', index=False)
    if 'FUNDING_CARRY_LOG' in globals() and not FUNDING_CARRY_LOG.empty:
        FUNDING_CARRY_LOG.to_excel(writer, 'Funding Carry Log', index=False)
    if 'BASIS_CARRY_LOG' in globals() and not BASIS_CARRY_LOG.empty:
        BASIS_CARRY_LOG.to_excel(writer, 'Basis Carry Log', index=False)
    if 'PAIR_STATARB_LOG' in globals() and not PAIR_STATARB_LOG.empty:
        PAIR_STATARB_LOG.to_excel(writer, 'Pair StatArb Log', index=False)
    if 'LIQUIDITY_REVERSAL_LOG' in globals() and not LIQUIDITY_REVERSAL_LOG.empty:
        LIQUIDITY_REVERSAL_LOG.to_excel(writer, 'Liquidity Reversal Log', index=False)
    if 'STOPLOSS_MOMENTUM_LOG' in globals() and not STOPLOSS_MOMENTUM_LOG.empty:
        STOPLOSS_MOMENTUM_LOG.to_excel(writer, 'Stoploss Momentum Log', index=False)
    if 'BETA_NEUTRAL_MOMENTUM_LOG' in globals() and not BETA_NEUTRAL_MOMENTUM_LOG.empty:
        BETA_NEUTRAL_MOMENTUM_LOG.to_excel(writer, 'Beta Neutral Momentum Log', index=False)

print('Saved V26 workbook:', xlsx_path)
print('Saved V26 metadata:', metadata_path)
print('Saved V26 daily returns:', returns_csv)
print('Saved V26 latest signal:', latest_signal_csv)

