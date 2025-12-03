export const BacktestRun = async (models, backtests, results) => {
  try {
    if (!models || models.length === 0) {
      console.warn("No models to run.");
      return;
    }
    if (!backtests || backtests.length === 0) {
      console.warn("No backtests to run.");
      return;
    }
    if (!results) {
      console.warn("No data provided.");
      return;
    }

    // Prepare payload
    const payload = {
      predictions: results,
      modelList: models,
      backtestList: backtests
    }

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

    const backtestResults = await response.json();
    // console.log(backtestResults);
    return backtestResults; // JSON results from backend
  } catch (err) {
    console.error("Error running backtests:", err);
    throw err;
  }
};
