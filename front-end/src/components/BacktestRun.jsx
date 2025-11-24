export const BacktestRun = async (models, backtests, rawData) => {
  try {
    if (!models || models.length === 0) {
      console.warn("No models to run.");
      return;
    }
    if (!backtests || backtests.length === 0) {
      console.warn("No backtests to run.");
      return;
    }
    if (!rawData) {
      console.warn("No data provided.");
      return;
    }

    // Prepare payload
    const payload = {
      rawData: rawData,
      modelList: models,
      backtestList: backtests,
      hyperparameters: {
        train_val_test_split: [0.8, 0.1, 0.1]
      }
    };

    const response = await fetch("http://localhost:5000/api/run_backtests", {
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
    console.error("Error running backtests:", err);
    throw err;
  }
};
