# Backend overview

This document summarises the backend of the Electronic Trading App. It is intended for developers who need to understand where responsibilities live and how the pieces fit together.

## High-level components

- `api_routes.py` — Flask endpoints used by the front-end. Entrypoints include data upload, fetching tickers, listing feature sets, and `POST /api/run_models` which triggers model runs.
- `process_data.py` — `DataProcessor` orchestrates feature engineering, splitting and windowed dataset creation.
- `window_generator.py` — `WindowGenerator` wraps TensorFlow's timeseries dataset creation and provides helpers for plotting and indexing.
- `training/runner.py` — `ModelRunner` coordinates running multiple models, groups by data config, caches processed data, and aggregates results.
- `training/trainer.py` — `ModelTrainer` handles per-model create/build/compile/train/evaluate lifecycle using the `ModelFactory` and `BaseModel` implementations.
- `models/` — model implementations and factory. Contains `implementations.py`, `factory.py`, and base classes.
- `model_handler.py` — legacy helper used for quick experiments and contains a variety of convenience model builders. Newer flow uses `training/runner.py` + `training/trainer.py`.
- `feature_engineer.py`, `data_splitter.py`, `ml_data.py`, `stock_data.py` — supporting utilities for feature creation, splitting and I/O.

## Design goals

- Separate data processing and model execution to avoid duplicate work when running multiple models on the same feature-set.
- Make the model run deterministic where possible (seed usage in model constructors).
- Provide a thin business-logic layer exposed via `/api/run_models` that performs validation and returns predictions + metrics.

For more detailed diagrams and the model-run sequence, see `api_flow.md` and `model_running.md`.