# RL Hybrid Maker/Taker para mercados binarios 5m

Proyecto modular de investigación para entrenar y evaluar agentes (baseline supervisado + RL DQN) sobre logs locales de microestructura y `cycle-summary`.

## Principios
- **Taker determinista**: buy al ask / sell al bid.
- **Maker conservador**: simulación probabilística con estados `pending/fill_favorable/fill_adverse/stale`.
- **Sin leakage**: split temporal por ciclo; features rolling causales.
- **Reproducible**: semillas y configs YAML.

## Estructura
- `src/rl_hybrid/data`: loaders, validación pydantic, alineación y checks de calidad.
- `src/rl_hybrid/features`: features causales, episodios y splits.
- `src/rl_hybrid/sim`: acciones, ejecución taker/maker y PnL.
- `src/rl_hybrid/env`: entorno Gymnasium multi-interacción.
- `src/rl_hybrid/models`: baseline tabular + red DQN + GRU supervisada.
- `src/rl_hybrid/train`: pipelines de preparación, supervisado y RL.
- `src/rl_hybrid/eval`: backtest y métricas.
- `configs/*.yaml`: configuración reproducible.
- `tests/`: unit e integración/smoke.

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Comandos
```bash
rlhyb prep-data --config configs/base.yaml
rlhyb train-sup --config configs/train_supervised.yaml
rlhyb train-rl --config configs/train_rl.yaml
rlhyb backtest --config configs/backtest.yaml
```

## Datos default
- `/mnt/data/microstructure-5m-1772372127598.jsonl`
- `/mnt/data/microstructure-5m-1772397985117.jsonl.gz`
- `/mnt/data/cycle-summary-5m-1772397985623.jsonl`

También se soporta `microstructure` en `.jsonl.zip` directamente (si contiene `.jsonl` o `.jsonl.gz` dentro), útil cuando subes los archivos comprimidos al repo.

## Convenciones implementadas
- `winner=null`: configurable con `exclude|truncate|keep`.
- Inventario configurable: `max_inventory_per_side`, `allow_dual_side_inventory`.
- Órdenes maker expiran por edad (`max_order_age`) y penalizan stale.
- Liquidación terminal: `UP` paga 1 si gana (0 si no); `DOWN` análogo.
- Maker no asume fills por touch: usa hazard/logística con regímenes `optimistic/base/pessimistic`.

## Limitaciones
- Sin matching engine ni prioridad de cola real.
- No se infiere tamaño real de fill por nivel (qty fija en esta baseline).
- Modelo maker explícitamente conservador para evitar sobre-optimismo.

## Métricas de evaluación
`reward_total`, `reward_mean`, `drawdown`, `profit_factor`, `hit_rate` + sensibilidad por régimen maker.
