# Regime Crypto Trader — V26 Long-Only

Production paper-trading repo for the V26 regime-gated, long-only crypto strategy.

## Strategy summary

The strategy is a deterministic risk-on crypto allocation model. It uses Alpaca daily crypto bars, computes BTC/ETH regime and breadth features, and only allocates when the V26 risk-on gate is true. When active, it ranks the supported crypto universe using momentum, volatility, moving-average, and stop-loss features, then creates long-only spot weights.

This repo is intentionally long-only because Alpaca crypto spot does not support short crypto or margin.

## Notional

Default notional is set to:

```text
MODEL_NOTIONAL=1000000
DEFAULT_ACCOUNT_VALUE=1000000
```

Target portfolio gross exposure is capped by `config/frozen_config.yaml` at 75 percent gross with a 20 percent single-asset cap.

## Required GitHub secrets

Add these in `regime_crypto_trader -> Settings -> Secrets and variables -> Actions -> Secrets`:

```text
APCA_API_KEY_ID
APCA_API_SECRET_KEY
```

Alternative supported names:

```text
ALPACA_API_KEY
ALPACA_SECRET_KEY
ALPACA_REGIME_CRYPTO_KEY_ID
ALPACA_REGIME_CRYPTO_SECRET_KEY
```

## GitHub variables

Add these in Actions variables:

```text
MODEL_NOTIONAL=1000000
DEFAULT_ACCOUNT_VALUE=1000000
EXECUTE_PAPER_ORDERS=false
SUBMIT_ORDERS=false
STRATEGY_KILL_SWITCH=false
```

Start with `EXECUTE_PAPER_ORDERS=false`. After planned orders look correct, switch to `true`.

## Canonical dashboard logs

The workflow writes QSentia-compatible logs:

```text
logs/portfolio/portfolio.csv
logs/decisions/latest_decision.csv
logs/decisions/decisions.csv
logs/target_weights/latest_target_weights.csv
logs/target_weights/target_weights.csv
logs/orders/latest_planned_orders.csv
logs/orders/planned_orders.csv
logs/orders/latest_submitted_orders.csv
logs/orders/submitted_orders.csv
logs/positions/latest_positions.csv
logs/positions/positions.csv
logs/health/signal_history.csv
logs/health/health_status.json
```

## Manual run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.template .env
python run_daily.py
```

## QSentia site registry entry

Add this to `Base_Model_BR_PPO/models.yaml`:

```yaml
  - id: regime_crypto_trader
    name: "Regime Crypto Trader — V26 Long-Only"
    description: "V26 regime-gated long-only crypto strategy. Uses Alpaca daily crypto bars, risk-on regime gating, momentum/volatility ranking, and QSentia-compatible paper trading logs. Default notional is 1000000."
    repo: "FinTechEntrepreneurldz/regime_crypto_trader"
    logs_path: "logs"
    branch: "main"
    enabled: true
    color: "#06b6d4"
```
