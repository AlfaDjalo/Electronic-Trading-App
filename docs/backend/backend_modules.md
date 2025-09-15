# Backend Modules

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
    A[Raw Data] --> B[DataProcessor]

    subgraph B[DataProcessor]
        B1[FeatureEngineering]
        B2[DataNormalization]
        B3[SplitData]
        B4[WindowGenerator]
    end

    A --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C[Train/Val/Test Windows]
    C --> D[Model Training]