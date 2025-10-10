# Backend Modules

## `model_service.py`

### `process_models_request()`
- **Purpose**: Main business logic function that processes ML requests from the API layer.
- **Parameters**:
  - `raw_data_list`: (list[dict]) - Time series data in API format (list of records)
  - `model_list`: (list[dict]) - Model configurations to run
  - `hyperparameters`: (dict) - Training parameters including train/val/test split
  - `feature_sets_file`: (str) - Path to feaure sets configuration
  - `verbose`: (bool, default: False) - Enable detailed logging
- **Returns**:
  - `dict` with keys: `success`, `dates`, `actual`, `predictions`, `stats`, `errors`, `metadata`
- **Functionality**
 - Input validation (data format, required fields, model list)
 - Data conversion (list -> Dataframe)
 - Creates `ModelRunner` with real components
 - Processes each model with error isolation
 - Aggregates results with metadata
- **Error Handling**: Graceful degradation - continues processing other models if one fails


## `training/runner.py`

### `ModelRunner`
- **Purpose**: Handles running multiple models with shared data processing.
- **Parameters**:
  - `raw_data`: (Pandas dataframe) - Dataframe of time series data
  - `feature_set_manager`: (FeatureSetManager object) - Object containing feature set dictionary
  - `hyperparameters`: (dict) - Training parameters including train/val/test split
  - `verbose`: (bool, default: False) - Enable detailed logging
- **Returns**:
  - `dict` with keys: `dates`, `actual`, `predictions`, `stats`
- **Functionality**
  - Manages cache of processed data
  - Creates `ModelConfig` with architecture and training parameters
  - Creates `ModelTrainer` 
  - Runs models


## `training/trainer.py`

### `ModelTrainer`
- **Purpose**: Handles model training and evaluations.
- **Parameters**:
  - `config`: (ModelConfig object) - Model configuration
  - `verbose`: (bool, default: False) - Enable detailed logging
- **Returns**:
  - fitted model
  - `dict` for model evaluation with keys: `metrics`, `actual`, `predictions`, `stats`
- **Functionality**
  - Creates model based on model config
  - Trains model
  - Evaluates model  






## `process_data.py`

### `DataProcessor`
- **Purpose**: Process raw time series data for model training.
  - Handles feature engineering, normalization, train/validation/test split, and prepares sliding windows for ML models.
- **Constructor (`__init__`) Parameters**:
  - `raw_data` (DataFrame | list of dicts) – Input raw data.
  - `feature_set` (dict) – Defines features to extract/transform.
  - `forecast_period` (int, default: 1) – How many steps ahead to forecast.
  - `normalise` (bool, default: True) – Whether to normalize numeric columns.
  - `train_ratio` (float, default: 0.8) – Fraction of data for training.
  - `val_ratio` (float, default: 0.1) – Fraction of data for validation.
- **State**:
  - `raw_df` (DataFrame) – Original input data.
  - `train_df`, `val_df`, `test_df` (DataFrame) – Split and normalized data.
  - `window` (WindowGenerator) – Sliding window generator.
- **Methods**:
  - `get_data()` → `(train_window, val_window, test_window)`  
    - Prepares numeric-only data and initializes the `WindowGenerator`.
    - Returns sliding window datasets for model consumption.
- **Calls**:
  - `FeatureEngineering` → Extract / select features.
  - `DataNormalization` → Scale numeric features.
  - `SplitData` → Partition into train / val / test sets.
  - `WindowGenerator` → Create sliding windows for ML models.
- **Output**: Data ready for ML model: train, validation, and test windows.

### `WindowGenerator`
- **Purpose**: Generate sliding windows for time series ML models.
- **Constructor (`__init__`) Parameters**:
  - `input_width` (int) – Number of time steps in input sequence.
  - `label_width` (int) – Number of time steps in output/label sequence.
  - `shift` (int) – Gap between input and label windows.
  - `train_df`, `val_df`, `test_df` (DataFrame) – Split data to generate windows from.
  - `label_columns` (list[str], optional) – Columns to predict; defaults to all numeric columns.
- **State**:
  - `train_df`, `val_df`, `test_df` (DataFrame) – Data slices.
  - `input_indices`, `label_indices` (np.ndarray) – Indices for slicing input and labels.
  - `column_indices` (dict) – Column name → index mapping.
- **Methods**:
  - `train` / `val` / `test` → `np.ndarray`
    - Returns numpy arrays with sliding windows for training, validation, or testing.
- **Output**: `np.ndarray` with shape `(num_windows, input_width, num_features)` for inputs and `(num_windows, label_width, num_features)` for labels.

## `FeatureEngineer`
- **Purpose**: Generate derived features from raw market data.
- **Inputs**: Raw DataFrame, feature set configuration
- **Outputs**: Feature-enhanced DataFrame
- **Notes**: Config-driven; supports multiple feature sets.

## `DataSplitter`
- **Purpose**: Split engineered DataFrame into train/val/test.
- **Inputs**: Processed DataFrame, ratios
- **Outputs**: `(train_df, val_df, test_df)`

## `Normalizer`
- **Purpose**: Scale numeric features using fitted statistics.
- **Inputs**: Train DataFrame
- **Outputs**: Normalized train/val/test sets
- **Notes**: Stores parameters for denormalization later.

## `WindowGenerator`
- **Purpose**: Convert normalized time series into sliding windows.
- **Inputs**: Train/val/test DataFrames, input/label widths, shift
- **Outputs**: `tf.data.Dataset` objects ready for model training
---

## Program Flow (Mermaid)
```mermaid
flowchart TD
    A[API Request] --> B[model_service.process_models_request]
    
    B --> C[Input Validation]
    C --> D[ModelRunner]
    
    subgraph D[ModelRunner - Batch Processing]
        D1[Group by Data Config]
        D2[Check Cache]
        D3[DataProcessor]
        D4[ModelTrainer]
    end
    
    D1 --> D2
    D2 -->|Cache Miss| D3
    D2 -->|Cache Hit| D4
    D3 --> D4
    
    subgraph D4[ModelTrainer]
        D4A[ModelFactory]
        D4B[Model.build/compile]
        D4C[Model.fit]
        D4D[Model.evaluate]
    end
    
    D4 --> E[Aggregate Results]
    E --> F[API Response]