# React Components

## `ModelSelect.jsx`
- **Purpose**: Allows user to select model, feature set, and params
- **Props**:
  - `onSubmit(modelConfig)`
- **State**:
  - `model`
  - `featureSet`
  - `params`
- **Output**: Calls API to trigger backend run

## `ChartArea.jsx`
- **Purpose**: Displays charted results
- **Props**:
  - `data`
- **State**:
  - `selectedFeatures`
- **Output**: Recharts graph