export const ProcessData = async (rawData, featureSets) => {
  try {
    if (!featureSet || featureSet.length === 0) {
      console.warn("No data to process.");
      return;
    }
    if (!rawData) {
      console.warn("No data provided.");
      return;
    }

    // Prepare payload
    const payload = {
      rawData,
      featureSets
    };
    // const payload = {
    //   models,       // list of model configurations
    //   data: dataInfo // the uploaded time series data
    // };

    const response = await fetch("http://localhost:5000/api/process_data", {
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
    console.error("Error processing data:", err);
    throw err;
  }
};
