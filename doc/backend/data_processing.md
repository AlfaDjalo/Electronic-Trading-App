# Data Processing Pipeline

This document describes the data processing pipeline orchestrated by `process_data.py` and the `DataProcessor` class. The pipeline prepares raw time series data for machine learning models through feature engineering, splitting, normalization, and windowing.

## Overview

The `DataProcessor` class handles the entire workflow from raw data to windowed TensorFlow datasets ready for model training. The pipeline is deterministic and reusable, allowing multiple models to share the same processed data.

## High-level process flow

```mermaid
flowchart TD
  A["Raw Time Series Data<br/>(DataFrame or list)"] --> B["Step 1: Feature Engineering<br/>(FeatureEngineer)"]
  B --> C["Engineered Features DataFrame<br/>(+ derived columns)"]
  C --> D["Step 2: Train/Val/Test Split<br/>(DataSplitter)"]
  D --> E["Three Split DataFrames<br/>(train_df, val_df, test_df)"]
  E --> F{"Normalise?"}
  F -->|Yes| G["Step 3a: Fit StandardScaler<br/>(on train data)"]
  F -->|No| H["Step 3b: Skip Normalisation"]
  G --> I["Apply Scaler to All Splits"]
  I --> J["Normalised DataFrames"]
  H --> J
  J --> K["Step 4: Windowing<br/>(WindowGenerator)"]
  K --> L["TensorFlow tf.data.Dataset<br/>(train, val, test batches)"]
  L --> M["Ready for Model Training"]
  
  style A fill:#e1f5ff
  style M fill:#c8e6c9
  style B fill:#fff9c4
  style D fill:#fff9c4
  style G fill:#fff9c4
  style K fill:#fff9c4
```

## Detailed step-by-step flow

### Step 1: Feature Engineering

```mermaid
flowchart LR
  A["Raw Data<br/>(price, volume, etc.)"] --> B["FeatureEngineer"]
  B --> C["Apply Feature Definitions<br/>(from feature_set config)"]
  C --> D["Compute Features<br/>(raw_data, lags, averages,<br/>RSI, Bollinger, etc.)"]
  D --> E["Engineered DataFrame<br/>(with all features + target)"]
  
  style A fill:#e1f5ff
  style E fill:#c8e6c9
```

**Inputs:**
- `raw_data`: DataFrame or dict list with OHLCV data
- `feature_set`: Configuration dict specifying which features to compute
  - Defined in `back-end/data/feature_sets.json`
  - Contains `features` list and optional `target` spec

**Outputs:**
- Engineered DataFrame with derived features (e.g., Close_lagged_1, RSI, Bollinger bands)
- Target column (usually `target` created from the target spec)

**Key code:**
```python
fe = FeatureEngineer(self.raw_df, feature_set)
self.df = fe.apply().dropna()
target_col = self.get_target()
```

### Step 2: Train/Val/Test Splitting

```mermaid
flowchart LR
  A["Engineered DataFrame<br/>(full time series)"] --> B["DataSplitter"]
  B --> C["Compute Split Indices<br/>(train_ratio, val_ratio)"]
  C --> D["Slice DataFrame<br/>80% train, 10% val, 10% test"]
  D --> E["train_df"]
  D --> F["val_df"]
  D --> G["test_df"]
  
  style A fill:#e1f5ff
  style E fill:#c8e6c9
  style F fill:#c8e6c9
  style G fill:#c8e6c9
```

**Inputs:**
- Engineered DataFrame
- `train_ratio` (default 0.8): proportion for training
- `val_ratio` (default 0.1): proportion for validation
- Remainder goes to test

**Outputs:**
- Three DataFrames: `train_df`, `val_df`, `test_df`

**Key code:**
```python
splitter = DataSplitter(self.df, train_ratio, val_ratio)
self.train_df, self.val_df, self.test_df = splitter.split()
```

### Step 3: Normalization (Optional)

```mermaid
flowchart LR
  A["train_df<br/>val_df<br/>test_df"] --> B{"normalise flag?"}
  B -->|False| C["Skip normalization"]
  B -->|True| D["Identify Numeric Columns"]
  D --> E["Fit StandardScaler<br/>(on train_df only)"]
  E --> F["Transform All Splits<br/>(using train statistics)"]
  F --> G["Normalized DataFrames"]
  C --> G
  
  style A fill:#e1f5ff
  style G fill:#c8e6c9
```

**Inputs:**
- `normalise` flag (boolean)
- Train, validation and test DataFrames

**Process:**
- If `normalise=True`:
  1. Fit `StandardScaler` on training data only (prevents data leakage)
  2. Transform validation and test using training statistics
  3. Store scaler for inverse transformation later (e.g., when displaying predictions)

**Outputs:**
- Normalized DataFrames (or unchanged if `normalise=False`)
- Scaler instance saved as `self.scaler` for inverse transforms

**Key code:**
```python
if self.normalise:
    numeric_cols = self.train_df.select_dtypes(include=["number"]).columns
    self.scaler = StandardScaler()
    self.scaler.fit(self.train_df[numeric_cols])
    self.train_df.loc[:, numeric_cols] = self.scaler.transform(...)
    # ... repeat for val and test
```

### Step 4: Windowing

```mermaid
flowchart LR
  A["train_df<br/>val_df<br/>test_df"] --> B["WindowGenerator"]
  B --> C["Compute Window Parameters<br/>(input_width, label_width,<br/>shift/forecast_period)"]
  C --> D["timeseries_dataset_from_array"]
  D --> E["split_window mapping<br/>(inputs, labels)"]
  E --> F["tf.data.Dataset<br/>train batches"]
  E --> G["tf.data.Dataset<br/>val batches"]
  E --> H["tf.data.Dataset<br/>test batches"]
  
  style A fill:#e1f5ff
  style F fill:#c8e6c9
  style G fill:#c8e6c9
  style H fill:#c8e6c9
```

**Inputs:**
- Normalized DataFrames
- `input_width`: number of timesteps to look back (e.g., 10)
- `forecast_period` (shift): forecast horizon (e.g., 1 day ahead)
- `label_width`: typically 1 (single-step prediction) or equals `forecast_period`

**Process:**
1. Compute total window size = `input_width + forecast_period`
2. Use `tf.keras.utils.timeseries_dataset_from_array` to create sliding windows
3. Map windows through `split_window()` to separate inputs and labels
4. Return as `train`, `val`, `test` properties

**Example:**
- If `input_width=10`, `forecast_period=1`:
  - Each window uses 10 timesteps as input
  - Predicts 1 timestep ahead
  - Window size = 11

**Outputs:**
- `tf.data.Dataset` objects ready for TensorFlow model training
- Batched into groups (default batch_size=32)

**Key code:**
```python
self.window = WindowGenerator(
    input_width=self.input_width,
    label_width=1,
    shift=self.forecast_period,
    train_df=self.train_df,
    val_df=self.val_df,
    test_df=self.test_df,
    label_columns=["target"]
)
```

## Complete pipeline sequence diagram

```mermaid
sequenceDiagram
  participant User
  participant DataProcessor
  participant FeatureEngineer
  participant DataSplitter
  participant StandardScaler
  participant WindowGenerator

  User->>DataProcessor: __init__(raw_data, feature_set, ...)
  DataProcessor->>FeatureEngineer: create and apply()
  FeatureEngineer-->>DataProcessor: engineered DataFrame
  DataProcessor->>DataSplitter: create and split()
  DataSplitter-->>DataProcessor: train_df, val_df, test_df
  
  alt normalise == True
    DataProcessor->>StandardScaler: fit(train_df)
    StandardScaler-->>DataProcessor: fitted scaler
    DataProcessor->>StandardScaler: transform(all splits)
    StandardScaler-->>DataProcessor: normalized splits
  end
  
  DataProcessor->>WindowGenerator: create with splits
  WindowGenerator-->>DataProcessor: window object
  
  User->>DataProcessor: get_data()
  DataProcessor-->>User: dict with train/val/test tf.data.Dataset + X/y splits
```

## API: `DataProcessor.get_data()`

The primary method to retrieve processed datasets:

```python
processed_data = data_processor.get_data(forecast_period=None)
```

**Returns dict with keys:**
- `'train'`, `'val'`, `'test'`: `tf.data.Dataset` objects (batched windows)
- `'x_train'`, `'y_train'`: training features (DataFrame) and target (Series)
- `'x_val'`, `'y_val'`: validation features and target
- `'x_test'`, `'y_test'`: test features and target

## Inverse transformation (denormalization)

After model training and prediction, denormalize predictions if needed:

```python
if data_processor.get_normalise():
    target = data_processor.get_target()
    y_pred_denorm = data_processor.inverse_transform(y_pred, target)
```

This uses the fitted scaler to reverse the normalization, so predictions are on the original scale.

## Caching and reuse

`ModelRunner` caches `DataProcessor` instances keyed by:
```python
config_key = (feature_set_name, forecast_period, input_width, normalise)
```

When multiple models use the same feature set and parameters, the cached `DataProcessor` is reused, avoiding duplicate feature engineering and windowing work.

## Common edge cases

1. **Short time series**: If the total number of samples is less than the total window size, `WindowGenerator` produces zero windows. Models should handle or error gracefully.

2. **Missing or invalid features**: If a feature specified in the feature set cannot be computed, `FeatureEngineer.apply()` may raise an error or produce NaN. Rows with NaN are dropped with `.dropna()`.

3. **Non-numeric columns**: Only numeric columns are normalized; categorical or string columns are excluded from the scaler.

4. **Date alignment**: `WindowGenerator` uses DataFrame indices; ensure dates are preserved as DataFrame indices (or separate column) if timestamps are needed for visualization.

## Configuration example

Feature set JSON (from `feature_sets.json`):

```json
{
  "Daily_data_historical": {
    "features": [
      {
        "name": "close",
        "function": "raw_data",
        "input_data_fields": ["close"]
      },
      {
        "name": "close_lag_1",
        "function": "create_lag",
        "input_data_fields": ["close"],
        "function_parameters": {"num_lags": 1}
      }
    ],
    "target": {
      "function": "raw_data",
      "input_data_fields": ["close"],
      "name": "target"
    },
    "data_type": "daily"
  }
}
```

Then:

```python
data_processor = DataProcessor(
    raw_data=df,
    feature_set=feature_set_manager.get_feature_set("Daily_data_historical"),
    forecast_period=1,
    input_width=10,
    normalise=True,
    train_ratio=0.8,
    val_ratio=0.1
)

processed_data = data_processor.get_data()
# processed_data['train'], processed_data['val'], processed_data['test'] are ready for models
```

