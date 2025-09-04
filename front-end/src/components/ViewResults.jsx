import { useState } from "react";
import { SeriesSelector } from "./SeriesSelector";
// import { DataSummary } from "./DataSummary";
import { ViewChart } from "./ViewChart";

export const ViewResults = ({ results }) => {
  if (!results || !results.dates) {
    return <p className="p-4">No results available yet.</p>
  }
  
  const { dates, actual, predictions, stats } = results;
  const modelNames = Object.keys(predictions)
  
  const [selectedSeries, setSelectedSeries] = useState(["Actual", ...modelNames]);
  const [viewMode, setViewMode] = useState("predictions");
  
  const chartData = dates.map((date, i) => {
    const row = {date};

    if (viewMode === "predictions") {
      if (i < actual.length) {
        row["Actual"] = Array.isArray(actual[i]) ? (Array.isArray(actual[i][0]) ? actual[i][0][0] : actual[i][0]) : actual[i];
      }
  
      modelNames.forEach(name => {
        if (i < predictions[name].length) {
          const val = predictions[name][i];
          row[name] = Array.isArray(val) ? (Array.isArray(val[0]) ? val[0][0] : val[0]) : val;
       }
      });
    } else if (viewMode === "errors") {
      modelNames.forEach(name => {
        if (i < predictions[name].length && i < actual.length) {
          const predVal = Array.isArray(predictions[name][i]) ? predictions[name][i][0] : predictions[name][i];
          const actVal = Array.isArray(actual[i]) ? actual[i][0] : actual[i];
          row[name] = predVal - actVal;
        }
      });
    }

    return row;
  });

  // console.log("chartData sample:", chartData[0]);


  // const [selectedSeries, setSelectedSeries] = useState(modelNames.length > 0 ? [modelNames[0]] : []);

  // console.log(dataInfo.feature_names)

  return (
    <div className="flex h-screen">

      {/* Left panel */}
      <div className="w-1/4 p-4 border-r border-gray-300">
        <button
          onClick={() => setViewMode(viewMode === "predictions" ? "errors" : "predictions")}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
        >
          {viewMode === "predictions" ? "Switch to Errors" : "Switch to Predictions"}
        </button>

        <SeriesSelector
          seriesNames={viewMode === "predictions" ? ["Actual", ...modelNames] : modelNames}
          selectedSeries={selectedSeries}
          onChange={setSelectedSeries}
        />
      </div>

      {/* Right panel */}
      <div className="w-3/4 p-4">
        <h2 className="text-lg font-bold mt-4">
          {viewMode === "predictions" ? "Forecasts" : "Prediction Errors"}
        </h2>
        <ViewChart
          chartData={chartData}
          selectedSeries={selectedSeries}
        />
      </div>
    </div>
  );
};

