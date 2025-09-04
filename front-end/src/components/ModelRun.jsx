export const ModelRun = async (models, rawData) => {
  try {
    if (!models || models.length === 0) {
      console.warn("No models to run.");
      return;
    }
    if (!rawData) {
      console.warn("No data provided.");
      return;
    }

    // console.log(rawData)

    // Prepare payload
    const payload = {
      rawData: rawData,  // rename "data" → "rawData"
      // rawData: dataInfo.timeSeriesData,  // rename "data" → "rawData"
      modelList: models,                 // rename "models" → "modelList"
      hyperparameters: {                 // optional, if you want to send splits etc
        train_val_test_split: [0.8, 0.1, 0.1]
      }
    };

    // const payload = {
    //   models,       // list of model configurations
    //   data: dataInfo // the uploaded time series data
    // };

    const response = await fetch("http://localhost:5000/api/run_models", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Server error: ${text}`);
    }

    const results = await response.json();
    return results; // JSON results from backend
  } catch (err) {
    console.error("Error running models:", err);
    throw err;
  }
};
