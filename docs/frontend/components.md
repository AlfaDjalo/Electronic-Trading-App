# React Components

## `ModelForm.jsx`
- **Purpose**: Allows user to select a single model, feature set, and params
- **Props**:
  - `modelConfig`: (type ? list of strings ?) - purpose list of available models ?
  - `featureSets`: ()
  - `initialValues`: (dictionary ?) - id is Add/Edit
  - `onSubmit`:
- **State**:
  - `selectedModel`: (string) - the model selected by the user using the dropdown
  - `selectedFeatureSet`:
  - `normalise`:
  - `params`:
  - `forecastPeriod`:
  - `inputWidth`:  
- **Other Variables**
  - `safeInitial`:
  - `parametersForModel`:
- **Functions**
  - `getAvailableFeatureSets`:
  - `handleModelChange`:
  - `handleParamChange`:
  - `handleSubmit`: handler for submission of form. Calls onSubmit prop with state information. Resets form if in Add mode.
- **Components**
  - Nil
- **Output**: Calls API to trigger backend run


## `ModelSelect.jsx`
- **Purpose**: Allows user to select multiple models to run on the chosen data set
- **Props**:
  - `modelConfig`: (type ?) - purpose
  - `featureSets`: 
  - `modelList`:
  - `onAddModel`:
  - `onEdit`:
  - `onDelete`:
  - `rawData`:
  - `setResults`:
- **State**:
  - `editingModel`: (`dict` with keys: `name`, `model`, `featureSet`, `forecastPeriod`, `inputWidth`, `normalise` ) - model object being edited
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