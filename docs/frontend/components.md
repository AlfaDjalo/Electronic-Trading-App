# React Components

## `LoadData.jsx`
- **Purpose**: This component allows the user to load data into the application, either by uploading a local CSV file or by fetching data directly from Yahoo Finance.
It manages drag-and-drop uploads, validates file type and size, shows progress indicators, and handles errors.
It also provides the option to select financial data categories and tickers for fetching remote data.

- **Props**:
  - `onUploadSuccess`: (function, optional) — callback invoked when data upload or fetch completes successfully, props (results, fileName).
  - `onUploadError`: (function, optional) — callback invoked when upload or fetch fails, props (errorMessage)
  - `uploadedFileName`: (string, optional) — optional prop to show an already-uploaded file name when the component loads.

- **State**:
  - `mode` (string, default: "csv") - current mode for file input, either "csv" or "yahoo".
  - `category` (string) - Yahoo data category (e.g. "us", "australian", "fx", "crypto").
  - `tickers` (list of string ?) - list of available ticker symbols for the selected category.
  - `ticker` (string) - selected ticker symbol.
  - `startDate` (?) — date range for Yahoo Finance data.
  - `endDate` (?) — date range for Yahoo Finance data.
  - `selectedFileName` (string) — File object for the uploaded CSV.
  - `uploadStatus`(string) — "idle", "uploading", "success", or "error".
  - `errorMessage` (string) — error message displayed if validation or upload fails.
  - `uploadProgress` (number ?) — numeric upload progress percentage.

- **Refs**:
- `fileInputRef` — reference to the hidden file input element, used to trigger file selection or clear it programmatically.

- **Main Functions**:
- `validateFile(fileName)` - Ensures the selected file is a .csv and under 10 MB. Returns an error string or null if valid.
- `handleFileSelect(file)` - Validates a dragged or chosen file, then triggers upload.
- `handleFileInputChange(event)` - Reads a file from the system file picker and validates it.
- `handleDragOver / handleDragEnter / handleDragLeave / handleDrop(event)` - Implements drag-and-drop behavior for the upload area. Only drop performs an action.
- `handleUpload(fileToUpload)` - Performs the upload via an XMLHttpRequest to the /api/upload_data endpoint, tracking progress and handling response callbacks.
- `handleClear()` - Clears uploaded file state, resets progress and errors.
- `handleYahooSubmit(event)` - Sends a POST request to /api/yahoo_data with the selected ticker and date range.
- `formatFileSize(bytes)` - Converts file size to a human-readable format.

- **UI Structure**:
- Mode selector buttons for CSV Upload and Yahoo Finance.
- When in CSV mode:
    Drag-and-drop upload box.
    Progress bar and file info.
    Validation and status display.

- When in Yahoo mode:
    Category selector.
    Ticker dropdown loaded from backend.
    Date range pickers.
    Fetch button for retrieving data.

- **Backend API Dependencies**:
- The component relies on the following endpoints:
- `POST /api/upload_data`: 
    Accepts multipart form-data with a file.
    Returns parsed time series data and optional feature sets.
    Example response:
    {
      "success": true,
      "timeSeriesData": [...],
      "featureSets": {...}
    }

- `POST /api/yahoo_data`:
    Accepts { ticker, start_date, end_date }.
    Fetches market data from Yahoo Finance.
    Example response:
    { 
      "success": true,
      "timeSeriesData": [...] 
    }

- `GET /api/get_tickers/<category>`:
    Returns available tickers for a given market category.
    Example response:
    { 
      "success": true,
      "tickers": ["AAPL", "MSFT", ...]
    }

- **Libraries and Hooks**:
  - `React`: useState, useRef, useEffect
  - `Native HTML`: drag-and-drop and file input APIs

- **Data Flow**:
- User selects upload mode.
- Upload or fetch is triggered.
- On success, onUploadSuccess sends parsed data back to the parent (App.jsx), which sets it as rawData.
- On failure, onUploadError handles the message and resets state.



## `ModelSelect.jsx`
- **Purpose**: Allows the user to define, edit, and manage multiple models to run on the uploaded dataset.
    It integrates the model creation form (ModelForm), displays a table of all configured models, and triggers model training and evaluation (ModelRun).
    Users can:
    - Add new models with chosen parameters and feature sets
    - Edit or delete existing models
    - Execute all configured models simultaneously and view the results

- **Props**:
  - `modelConfig`: (object) — configuration object listing available models and their parameter definitions, passed down to ModelForm.
  - `featureSets`: (object) — available feature sets generated from uploaded data.
  - `modelList`: (array) — list of current models added by the user, each including fields such as name, model type, feature set, forecast period, input width, and normalisation flag.
  - `onAddModel`: (function) — callback to add a new model to the global list. Triggered by successful form submission in Add mode.
  - `onEdit`: (function) — callback to update an existing model after editing.
  - `onDelete`: (function) — callback to remove a model from the list.
  - `rawData`: (array/object) — uploaded dataset used as input for training and testing.
  - `setResults`: (function) — callback to store model run results.
- **State**:
  - `editingModel`: (`dict` with keys: `name`, `model`, `featureSet`, `forecastPeriod`, `inputWidth`, `normalise` ) - holds the currently selected model being edited, or null when adding a new one.
    Expected keys:
    { id, name, model, featureSet, forecastPeriod, inputWidth, normalise, params }

- **Hooks**:
  - `useState` — manages edit mode (selected model for editing).
  - `useNavigate` (from react-router-dom) — used to navigate programmatically to the /view_results page after running models.

- **Main functions**:
  - `handleAdd(newModel)` - Creates a unique id for the new model (if not already provided), calls onAddModel, and resets editingModel to null.
  - `handleUpdate(updatedModel)` - Updates an existing model by calling onEdit(updatedModel) and clears edit mode.
  - `onDelete(modelId)` - (prop callback) Removes the specified model from the list (handled in parent).
  - `ModelRun(modelList, rawData)` - Imported function that executes all models against the dataset and returns results for display in ViewResults.


- **Local Components**:
  - `ModelForm`: handles model parameter entry and validation
  - `ModelRun`: backend API interface for executing model training and evaluation

- **Libraries and Components**:
  - `React`: useState
  - `React Router`: useNavigate

- **Interactions & Data Flow**:
  - User fills out the ModelForm and submits.
    → handleAdd() or handleUpdate() updates the modelList.
  - The table reflects current models and allows further editing or deletion.
  - When ready, the user clicks Run Models.
    → ModelRun() runs all defined models, collects metrics and predictions.
    → Results are stored via setResults() and the app navigates to the results view.
  - ViewResults displays comparative performance and prediction charts.

- **UI Structure**:
  - `Model Form Section`:
    Uses <ModelForm> to define parameters for a new model or edit an  existing one.
    Form fields depend on modelConfig and featureSets props.
  - `Existing Models Table`:
    Displays a summary of all configured models in tabular format, with columns:
    - Name
    - Model Type
    - Feature Set
    - Forecast Period
    - Input Width
    - Normalise (Yes/No)
    - Actions (Edit/Delete)
  - `Run Models Button`:
    Appears below the table when one or more models exist.
    Clicking it triggers:
    - ModelRun(modelList, rawData)
    - Saves results via setResults
    - Navigates to /view_results

- **Output**: Calls API to trigger backend run
  - Updates the global model list in parent state (App.jsx).
  - On Run: executes all models via backend API and passes results to ViewResults.jsx.



## `ModelForm.jsx`
- **Purpose**: Allows user to select a single model, feature set, and params
  Provides a user interface for defining a single predictive model configuration.
  The form allows users to select a model type, associated feature set, forecast period, input width, and normalisation setting, as well as adjust model-specific parameters dynamically based on the selected model.
  It supports both “Add” and “Edit” modes depending on whether initialValues includes an id.

- **Props**:
  - `modelConfig`: (object) — dictionary of available models and their configurable parameters.
    Example:
    {
      "linear_regression": { 
        "learning_rate": { default: 0.01 }, "fit_intercept": { default: true }
      },
      "lstm": {
        "hidden_size": { default: 64 }, "epochs": { default: 50 }
      }
    }
  - `featureSets`: (array) — list of available feature sets generated from uploaded data.
    Each feature set is expected to include at least { name: string }.
  - `initialValues`: (object, optional) — existing model configuration used to prefill the form during edit mode.
    Keys typically include:
    { id, name, model, featureSet, normalise, forecastPeriod, inputWidth, params }
  - `onSubmit`: (function) — callback executed when the form is submitted. Receives a fully constructed model configuration object.

- **State**:
  - `selectedModel`: (string) — currently selected model type from modelConfig.
  - `selectedFeatureSet`: (string) — currently chosen feature set name.
  - `normalise`: (boolean) — indicates whether input data should be normalised.
  - `params`: (object) — key–value pairs representing model-specific hyperparameters.
  - `forecastPeriod`: (number) — the forward-looking period (t + k) for prediction.
  - `inputWidth`: (number) — number of lagged input observations used to form a training window.

- **Derived Variables**
  - `safeInitial`: ensures initialValues is non-null to prevent runtime errors.
  - `parametersForModel`: dynamically extracts the parameter definitions for the selected model.
  - `getAvailableFeatureSets(selectedModel)`: filters feature sets depending on model type.
    - For 'baseline' models → returns only those starting with "Baseline_".
    - For all other models → excludes "Baseline_" feature sets.

- **Hooks**:
  - `useState` — manages all form input states.
  - `useEffect` — updates local form state when initialValues, featureSets, or modelConfig change, allowing live editing and refresh.

- **Functions**
  - `handleModelChange(newModel)` — updates selectedModel and resets the feature set list based on model type.
  - `handleParamChange(key, value)` — updates the parameter value for a given key within params.
  - `handleSubmit(event)` — main form submission handler.
    - Prevents default form behavior.
    - Constructs a model object combining:
    - modelConfig[selectedModel] defaults
    - User-provided parameter overrides (params)
    - Global form fields (featureSet, normalise, forecastPeriod, inputWidth)
    - Generates a model name as "<model>-<featureSet>" if not provided.
    - Calls onSubmit(data) with the full model definition.
    - Resets form to defaults when adding a new model (no id present).

- **Components**
  - Nil

- **UI Structure**:
  Left Column (General Settings):
    - Dropdown for selecting model type.
    - Dropdown for selecting feature set (filtered by model type).
    - Numeric input for Forecast Period (1–365).
    - Numeric input for Input Width (1–365).
    - Checkbox for Normalise flag.
  Right Column (Model Parameters):
    - Dynamically renders input fields for each parameter defined in modelConfig[selectedModel].
    - Each parameter’s default value comes from the configuration, with user overrides stored in params.
  Bottom Section (Submit Button):
    - Displays “Add Model” or “Update Model” depending on initialValues.id.
    - Calls onSubmit with form data when clicked.

- **Interactions & Data Flow**:
  1. When the user selects a model, its parameter fields are generated dynamically.
  2. The feature set dropdown adapts automatically for baseline or other model types.
  3. On form submission:
    - Parameters are merged with defaults.
    - The resulting configuration object is sent upward via onSubmit.
  4. Parent (ModelSelect.jsx) adds or updates this model in modelList.

- **Libraries and Components**:
  - React: useState, useEffect
  - Native HTML: <form>, <select>, <input> elements
  - Tailwind CSS classes for layout and styling

- **Output**: Calls API to trigger backend run
    Output Example (onSubmit payload):
    {
      id: 1730001234567,
      name: "lstm-FeatureSet_A",
      model: "lstm",
      featureSet: "FeatureSet_A",
      normalise: true,
      params: { hidden_size: 128, epochs: 50 },
      forecastPeriod: 10,
      inputWidth: 30
    }


## `ViewResults.jsx`
- **Purpose**: This component displays model performance results — including prediction accuracy metrics, forecast charts, and error visualizations — after one or more models have been trained.
  It integrates with SeriesSelector for selecting visible data series and ViewChart for rendering time-series plots. It can optionally reconstruct true calendar dates from the original dataset for accurate chart labeling.


- **Props**:
  - `results`: (object) — the results object returned from the backend after training/evaluation, containing:
    - dates: mapping of model → array of x-axis indices or date strings
    - actual: dictionary of actual (ground truth) values
    - predictions: dictionary of model → array of predicted values
    - stats: dictionary of model → metrics (MSE, MAE, RMSE, R², etc.)
    - metadata: optional object with info such as dataset splits
  
  - `rawdata`: (array, optional) - the raw uploaded dataset, used to reconstruct actual date labels for the test set when indices are numeric.

- **State**:
  - `selectedSeries` (array of strings) — currently visible time series on the chart (defaults to ["Actual", ...modelNames]).
  - `viewMode` (string) — toggles between "predictions" (forecasts vs. actual) and "errors" (prediction residuals).

- **Main Functions**:
  - `reconstructTestDates(rawData, split)` - Utility import that rebuilds a date series for the test set based on the dataset length and train/val/test proportions.
  Called once at initialization if rawData is provided.
  - `flattenedActual (derived variable)` - Converts the possibly nested actual data format into a simple mapping { dateIndex: value }.
  - `chartData (derived variable)` - Constructs an array of objects, one per date, each containing:
    - date: true date string (if reconstructed) or numeric fallback
    - Actual: observed ground truth
    - each model’s prediction or error (depending on viewMode)
      This array is passed to ViewChart.
  - `handleViewToggle()` - Toggles between "predictions" and "errors" modes to switch chart context.


- **UI Structure**:
  1. Statistics Table (Top Section)
    Displays model performance metrics in a grid:
    - Model name
    - MSE
    - MAE
    - RMSE
    - R² (formatted as a percentage)
    - Custom error messages if provided by the backend
  2. Main Layout (Two Columns)
    - Left Panel:
      - Toggle button for Predictions / Errors view
      - SeriesSelector component for toggling visibility of individual models and the actual series
    - Right Panel:
      - Chart area rendered by ViewChart
      - Displays either “Forecasts” or “Prediction Errors” based on viewMode


- **Backend Data Dependency**:
  This component expects to receive a standardized results object (typically from /api/train_models or similar) with this structure:
    {
      "dates": { "baseline": [0, 1, 2, ...] },
      "actual": { "0": [1.2], "1": [1.3], ... },
      "predictions": { "baseline": [[1.2], [1.4], ...] },
      "stats": { "baseline": { "mse": 0.01, "r2": 0.98 } },
      "metadata": { "split": [0.8, 0.1, 0.1] }
    }

- **Libraries and Hooks**:
  - React: useState
  - Local components:
    - SeriesSelector — allows users to pick which models or series appear on the chart.
    - ViewChart — renders the visual chart (likely using Recharts or Chart.js).
  - Utilities:
    - reconstructTestDates — reconstructs date labels for the test set.

- **Data Flow**:
  1. The parent (e.g. App.jsx or TrainModels.jsx) calls the backend and passes the response into ViewResults as results.
  2. If rawData is available, test dates are reconstructed for accurate chart labeling.
  3. The statistics table is rendered.
  4. The user can toggle between predictions and errors and select which model series to display.
  5. The chart updates dynamically based on the selected series and view mode.


## `SeriesSelector.jsx`
- **Purpose**: This component allows the user to choose which data series to display in plots or analyses.
    It presents a multi-select dropdown built with react-select, populated from the available series names.
    At least one series is always selected to ensure valid visualization behavior.

- **Props**:
  - `seriesNames` (array of string) — list of all available data series names that can be selected.
  - `selectedSeries` (array of string) — currently selected series.
  - `onChange` (function) — callback fired when the user changes their selection. Receives an array of selected series names.

- **State**:
  This component does not maintain internal state (besides internal react-select handling).
  All state is controlled externally via props and onChange.

- **Effects**:
  - `useEffect([...])` — ensures that when the component first loads, or when available options change, at least one series is selected.
    If selectedSeries is empty but options exist, it automatically selects the first available series.

- **Main Functions**:
  - The component uses onChange(selected) to pass back the updated list of selected series values to the parent component.
  - Series options are mapped to { value, label } objects for use with react-select.

- **UI Structure**:
  - A container with light gray background (bg-gray-100) and vertical scroll enabled (overflow-y-auto).
  - A heading: “Select Series”.
  - A multi-select dropdown from react-select, allowing multiple selections and search functionality.
  - Default styling is overridden for light theme readability (white menu, black text).

- **Behavior**:
  - Multi-select dropdown supports selecting or deselecting multiple data series.
  - Always enforces at least one selection (auto-selects the first available series if none are chosen).
  - Calls onChange whenever the user updates selections.

- **Libraries and Hooks**:
  - `react`: useEffect
  - `react-select`: multi-select dropdown component

- **Data Flow**:
  - seriesNames (list of available series) → transformed into dropdown options.
  - User changes selection → onChange called → parent updates selectedSeries → re-render with new values.

## `ChartArea.jsx`
- **Purpose**: This component is responsible for visualizing time series data and model predictions.
It renders an interactive line chart that displays the raw data, model outputs, and optionally multiple selected series.
    The chart automatically updates when new data or models are selected and can optionally show train/validation/test regions.
    It also manages date formatting and x-axis labeling logic.
- **Props**:
  - `chartData`: (array of objects) — combined dataset for charting, where each object represents one time step with named series values.
  - `selectedSeries`: (array of strings) — list of series names currently selected for display.
  - `testDates`: (array of strings, optional) — actual date labels for test data (used for x-axis if available).
  `showLegend`: (boolean, optional, default: true) — whether to display the legend.
  `height`: (number, optional) — chart height in pixels.
  `width`: (number, optional) — chart width in pixels.
  `title`: (string, optional) — chart title to display above the visualization.
- **State**:
  - (Typically stateless; depends on props from parent)
    The component itself does not maintain local React state, but it reacts to prop updates (e.g., when user selects different series or new results are loaded).
- **Refs**:
  - None (chart rendering handled directly through the Recharts library).
- **Main Functions**:
  - `formatXAxis(tickIndex)` — Formats x-axis labels.
Uses actual dates from testDates if available; otherwise falls back to numeric indices.
      Prevents default display of placeholder formats like "338-1-1".
  - `generateLines()` — Dynamically creates one <Line> element for each selected series, assigning unique colors and labels.
  - `handleTooltip(data)` — Custom tooltip formatter showing value and series name on hover.
- **UI Structure**:
  - Container <div> wrapping the chart with optional title and styling.
  - Uses Recharts components:
    - <ResponsiveContainer> for auto-scaling.
    - <LineChart> as the main chart container.
    - <CartesianGrid> for background grid lines.
    - <XAxis> for time or index labels (formatted using formatXAxis).
    - <YAxis> for numerical values.
    - <Tooltip> for value display on hover.
    - <Legend> to identify each plotted series.
    - One <Line> per selected series, styled with unique color and smooth interpolation.

- **Backend API Dependencies**:
  - None directly.
    The component is purely a frontend visualization layer that consumes data already fetched by other components (e.g., from /api/run_models).

- **Libraries and Hooks**:
  - `React`: functional component.
  - `Recharts`: LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer.

- **Data Flow**:
  1. The parent component (e.g., ViewResults.jsx) passes chartData, selectedSeries, and optional testDates to ViewChart.
  2. ViewChart builds a unified dataset suitable for Recharts, mapping each selected series to a <Line>.
  3. When the parent updates selectedSeries or new model results are loaded:
    - The chart re-renders automatically with updated lines.
  4. If actual date labels are available:
    - The x-axis shows real timestamps.
    - Otherwise, it displays simple numeric indices.



## `ModelRun.js`
- **Purpose**: This module provides a utility function to send model execution requests to the backend.
  It takes a list of model configurations and raw input data, sends them to the API, and returns the model run results.
  It is typically invoked when the user runs selected models from the interface (e.g., after data upload and selection).

- **Function**: - `ModelRun(models, rawData)`

- **Parameters**:
  - `models` (array of objects) — list of model configurations to execute. Each model entry typically includes model type, parameters, and metadata.
  - `rawData` (object or array) — time series or structured input data to train and test models on.

- **Returns**: A JSON response object from the backend containing model predictions, metrics, and metadata.

- **Internal Logic**:
  1. Validation:
    - Warns and exits early if no models are provided.
    - Warns and exits early if no data is available.
  2. Payload Preparation:
    - Constructs a JSON payload with:
      - rawData: the uploaded or fetched data.
      - modelList: the list of model definitions.
      - hyperparameters: optional extra parameters (e.g., train/validation/test split ratios).
  3. API Request:
    - Sends a POST request to the backend endpoint http://localhost:5000/api/run_models.
    - Uses JSON headers and stringified payload.
  4. Error Handling:
    - Checks response status and throws a descriptive error if the request fails.
    - Catches and logs runtime or network errors.
  5. Return Value:
    - Returns the parsed JSON body from the backend on success.

- **Backend API Dependencies**:
  - POST /api/run_models
    - Purpose: Executes one or more models on the provided dataset.
    - Request Body Example:
      {
        "rawData": [...],
        "modelList": ["LSTM", "Transformer"],
        "hyperparameters": {
          "train_val_test_split": [0.8, 0.1, 0.1]
        }
      }
    - Response Example:
      {
        "success": true,
        "results": [
          {
            "model": "LSTM",
            "predictions": [...],
            "metrics": {"RMSE": 0.012, "MAE": 0.009}
          }
        ]
      }

- **Error Cases**:
  - Missing or empty model list → logs "No models to run."
  - Missing data → logs "No data provided."
  - Server returns non-OK response → throws "Server error: <details>"

- **Libraries Used**:
  - `Fetch API` — for HTTP POST request
  - `JavaScript (ES6)` — async/await, try/catch

- **Data Flow**:
  - User triggers model run in frontend →
  - ModelRun() constructs payload →
  - Sends to backend /api/run_models →
  - Receives structured results →
  - Returns results to caller (e.g., for display in ViewResults.jsx).


## `ViewData.jsx`
- **Purpose**: This component provides a simple interface for exploring and visualizing uploaded or fetched raw data before modeling.
    It allows the user to select one or more data series from the dataset and view them in an interactive line chart.
    ViewData is primarily a data exploration and preview tool — it helps confirm that the dataset loaded correctly and gives a quick visual overview of trends and relationships between series.

- **Props**:
  - `data`: (array of objects) — time series data to visualize. Each object represents one time step, with keys for each series and a date or index field.
  - `seriesNames`: (array of strings) — list of available data series that can be plotted (e.g., columns in the CSV file or returned dataset).

- **State**:
  - `selectedSeries` (array of strings) — list of currently selected series to display on the chart.
    - Initialized to include the first series by default (if more than one is available).
    - Updated when the user selects or deselects items in the SeriesSelector.

- **Refs**:
  - None (selection and chart rendering are managed via React state and props).

- **Main Functions**:
  - (Stateless rendering component — no internal functions besides React hooks)
  - React’s useState manages which series are currently selected for display.

- **UI Structure**:
  - Left panel (Series Selector):
    - Uses the SeriesSelector component to list available series as checkboxes or multi-select dropdown options.
    - Updates selectedSeries state on user interaction.
    - Fixed-width panel with scrollable content and border.
  - Right panel (Chart View):
    - Uses the ViewChart component to plot the currently selected series.
    - The chart automatically updates whenever selectedSeries changes.
  - A simple message ("No data loaded yet.") is displayed if data is empty or undefined.
  - (Optional) — A placeholder for a future <DataSummary> component is included but currently commented out.

- **Backend API Dependencies**:
  - None directly.
    This component assumes that data has already been loaded and preprocessed (e.g., by LoadData.jsx or a parent component).
    It purely displays data passed via props.

- **Libraries and Hooks**:
  - React: useState
  - Local components:
    - SeriesSelector — for choosing which data series to display.
    - ViewChart — for rendering the actual line chart visualization.

- **Data Flow**:
  1. The parent component (e.g., App.jsx or a data upload handler) provides data and seriesNames.
  2. When ViewData renders:
    - It initializes the default selected series.
  3. The user chooses which series to visualize using SeriesSelector.
  4. The selectedSeries array updates in React state.
  5. The ViewChart component re-renders with the updated selection to display the corresponding lines.



