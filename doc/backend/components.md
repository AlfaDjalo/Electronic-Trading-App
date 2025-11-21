# Backend Components (detailed)

This document describes the purpose and responsibilities of each backend file relevant to model running.

## `api_routes.py`
- Purpose: Flask endpoints used by the front-end.
- Responsibilities:
  - Validate incoming requests (upload, run_models)
  - Load feature set definitions
  - Call business-logic layer (`services.model_service.process_models_request`)
  - Format and return JSON responses

## `process_data.py` (`DataProcessor`)
- Purpose: Orchestrate feature engineering, splitting, normalisation and windowing.
- Responsibilities:
  - Apply `FeatureEngineer` to raw data
  - Split into train/val/test using `DataSplitter`
  - Fit `StandardScaler` (if normalise=True) and transform splits
  - Create `WindowGenerator` with the requested `input_width` and `forecast_period`
  - Provide helpers: `get_data()`, `get_target()`, `get_label_timestamps()`, `inverse_transform()`

## `window_generator.py` (`WindowGenerator`)
- Purpose: Build TF `tf.data.Dataset` windows for model training and evaluation.
- Responsibilities:
  - Compute input/label slices
  - Convert DataFrame(s) into tf.keras timeseries_dataset_from_array
  - Provide `train`, `val`, `test` properties
  - Provide `example` batch for plotting/debugging

## `training/runner.py` (`ModelRunner`)
- Purpose: Coordinate running multiple models efficiently.
- Responsibilities:
  - Group models by data configuration to share processed data
  - Cache `DataProcessor` instances
  - Instantiate `ModelTrainer` per model and aggregate results

## `training/trainer.py` (`ModelTrainer`)
- Purpose: Create, build, compile, fit and evaluate a single model.
- Responsibilities:
  - Use `ModelFactory` to instantiate model classes
  - Train with TF datasets from `DataProcessor.window`
  - Evaluate on test dataset and compute metrics (MSE, MAE, RMSE, R2)
  - Provide helpers to show model structure and weights

## `models/factory.py` and `models/implementations.py`
- Purpose: Provide pluggable model implementations and a factory to create them by name.
- Notes: Add or register new models in `ModelFactory.register_model`.

## `model_handler.py`
- Purpose: Legacy helper offering a number of model-building utilities and LOB-specific helpers. Useful as reference and for experimenting with non-factory code paths.

## `services/model_service.py`
- Purpose: Business-logic glue used by `api_routes.py` — perform validation, call `ModelRunner`, format full response (metadata + aggregated results).


