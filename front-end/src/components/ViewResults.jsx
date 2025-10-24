import { useState } from "react";
import { SeriesSelector } from "./SeriesSelector";
// import { DataSummary } from "./DataSummary";
import { ViewChart } from "./ViewChart";

export const ViewResults = ({ results }) => {

  if (!results || !results.dates || Object.keys(results.dates).length === 0) {
    return <p className="p-4">No results available yet.</p>
  }
  
  const { dates, actual, predictions, stats } = results;
  const modelNames = Object.keys(predictions)
  const [selectedSeries, setSelectedSeries] = useState(["Actual", ...modelNames]);
  const [viewMode, setViewMode] = useState("predictions");
  
  // Build combined date set from all models
  const allDatesSet = new Set();
  modelNames.forEach(name => {
    if (dates[name]) {
      dates[name].forEach(date => allDatesSet.add(date));
    }
  });
  const allDates = Array.from(allDatesSet).sort();
  
  console.log("actual (raw):", actual);
  console.log("dates:", dates);
  console.log("predictions:", predictions);

  // 🧩 --- FLATTEN actual values so they match prediction format ---
  const flattenedActual = {};
  Object.keys(actual).forEach(key => {
    const value = actual[key];
    if (Array.isArray(value)) {
      // handle [[x]] or [x] shapes
      flattenedActual[key] = Array.isArray(value[0]) ? value[0][0] : value[0];
    } else {
      flattenedActual[key] = value;
    }
  });
  console.log("flattenedActual:", flattenedActual);

  // Create chartData with proper alignment
  const chartData = allDates.map(date => {
    const row = { date };

    if (viewMode === "predictions") {
      modelNames.forEach(name => {
        if (dates[name]) {
          const idx = dates[name].indexOf(date);
          if (idx >= 0 && idx < predictions[name].length) {
            const val = predictions[name][idx];
            row[name] = Array.isArray(val) 
              ? (Array.isArray(val[0]) ? val[0][0] : val[0]) 
              : val;
          }
        }

        // ✅ use flattenedActual instead of nested arrays
        if (flattenedActual[date] !== undefined) {
          row["Actual"] = flattenedActual[date];
        }
      });
    } else if (viewMode === "errors") {
      modelNames.forEach(name => {
        if (dates[name]) {
          const idx = dates[name].indexOf(date);
          if (idx >= 0 && idx < predictions[name].length && flattenedActual[date] !== undefined) {
            const predVal = Array.isArray(predictions[name][idx]) 
              ? predictions[name][idx][0] 
              : predictions[name][idx];
            const actVal = flattenedActual[date];
            row[name] = predVal - actVal;
          }
        }
      });
    }

    return row;
  });

  return (
    <div>
      <div>
        <h3>Model Statistics</h3>
        <table className="w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-100 text-black">
              <th className="border p-2">Model Name</th>
              <th className="border p-2">MSE</th>
              <th className="border p-2">MAE</th>
              <th className="border p-2">RMSE</th>
              <th className="border p-2">R²</th>
              <th className="border p-2">Error</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(results.stats).map(([modelName, stats]) => (
              <tr key={modelName}>
                <td className="border p-2">{modelName}</td>
                <td className="border p-2">
                  {typeof stats.mse === "number" ? stats.mse.toFixed(2) : stats.mse ?? "-"}
                </td>
                <td className="border p-2">
                  {typeof stats.mae === "number" ? stats.mae.toFixed(2) : stats.mae ?? "-"}
                </td>
                <td className="border p-2">
                  {typeof stats.rmse === "number" ? stats.rmse.toFixed(2) : stats.rmse ?? "-"}
                </td>
                <td className="border p-2">
                  {typeof stats.r2 === "number"
                    ? `${(stats.r2 * 100).toFixed(2)}%`
                    : stats.r2 ?? "-"}
                </td>
                <td className="border p-2">{stats.error ?? ""}</td>
              </tr>            
            ))}
          </tbody>
        </table>
      </div>

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
    </div>
  );
};




// import { useState } from "react";
// import { SeriesSelector } from "./SeriesSelector";
// // import { DataSummary } from "./DataSummary";
// import { ViewChart } from "./ViewChart";

// export const ViewResults = ({ results }) => {
//   // if (!results || !results.dates) {
//   //   return <p className="p-4">No results available yet.</p>
//   // }
  
//   if (!results || !results.dates || Object.keys(results.dates).length === 0) {
//     return <p className="p-4">No results available yet.</p>
//   }
  
//   const { dates, actual, predictions, stats } = results;
//   const modelNames = Object.keys(predictions)
//   const [selectedSeries, setSelectedSeries] = useState(["Actual", ...modelNames]);
//   const [viewMode, setViewMode] = useState("predictions");
  
//   // Build combined date set from all models
//   const allDatesSet = new Set();
//   modelNames.forEach(name => {
//     if (dates[name]) {
//       dates[name].forEach(date => allDatesSet.add(date));
//     }
//   });
//   const allDates = Array.from(allDatesSet).sort();
  
//   console.log("actual:", actual);
//   console.log("dates:", dates);
//   console.log("predictions:", predictions);

//   // Create chartData with proper alignment
//   // const chartData = allDates.map(date => {
//   //   const row = { date };
    
//   //   // Single actual value per date
//   //   if (actual && actual[date] !== undefined) {
//   //     row["Actual"] = actual[date];
//   //   }
    
//   //   // Per-model predictions
//   //   modelNames.forEach(name => {
//   //     if (dates[name]) {
//   //       const idx = dates[name].indexOf(date);
//   //       if (idx >= 0 && idx < predictions[name].length) {
//   //         // ... prediction mapping
//   //       }
//   //     }
//   //   });
    
//   //   return row;
//   // });
//   const chartData = allDates.map(date => {
//     const row = { date };

//     if (viewMode === "predictions") {
//       // For each model, find the date index in that model's date array
//       modelNames.forEach(name => {
//         if (dates[name]) {
//           const idx = dates[name].indexOf(date);
//           if (idx >= 0 && idx < predictions[name].length) {
//             const val = predictions[name][idx];
//             row[name] = Array.isArray(val) 
//               ? (Array.isArray(val[0]) ? val[0][0] : val[0]) 
//               : val;
//           }
//         }
        
//         // Add actual value for this model
//         if (actual[name]) {
//           const idx = dates[name].indexOf(date);
//           if (idx >= 0 && idx < actual[name].length) {
//             const actVal = actual[name][idx];
//             row["Actual"] = Array.isArray(actVal) 
//               ? (Array.isArray(actVal[0]) ? actVal[0][0] : actVal[0]) 
//               : actVal;
//             console.log(row["Actual"]);
//           }
//         }
//       });
//     } else if (viewMode === "errors") {
//       modelNames.forEach(name => {
//         if (dates[name]) {
//           const idx = dates[name].indexOf(date);
//           if (idx >= 0 && idx < predictions[name].length && idx < actual[name].length) {
//             const predVal = Array.isArray(predictions[name][idx]) ? predictions[name][idx][0] : predictions[name][idx];
//             const actVal = Array.isArray(actual[name][idx]) ? actual[name][idx][0] : actual[name][idx];
//             row[name] = predVal - actVal;
//           }
//         }
//       });
//     }

//     return row;
//   });

  
//   // const chartData = dates.map((date, i) => {
//   //   const row = {date};

//   //   if (viewMode === "predictions") {
//   //     if (i < actual.length) {
//   //       row["Actual"] = Array.isArray(actual[i]) ? (Array.isArray(actual[i][0]) ? actual[i][0][0] : actual[i][0]) : actual[i];
//   //     }
  
//   //     modelNames.forEach(name => {
//   //       if (i < predictions[name].length) {
//   //         const val = predictions[name][i];
//   //         row[name] = Array.isArray(val) ? (Array.isArray(val[0]) ? val[0][0] : val[0]) : val;
//   //      }
//   //     });
//   //   } else if (viewMode === "errors") {
//   //     modelNames.forEach(name => {
//   //       if (i < predictions[name].length && i < actual.length) {
//   //         const predVal = Array.isArray(predictions[name][i]) ? predictions[name][i][0] : predictions[name][i];
//   //         const actVal = Array.isArray(actual[i]) ? actual[i][0] : actual[i];
//   //         row[name] = predVal - actVal;
//   //       }
//   //     });
//   //   }

//   //   return row;
//   // });

//   // console.log("chartData sample:", chartData[0]);


//   // const [selectedSeries, setSelectedSeries] = useState(modelNames.length > 0 ? [modelNames[0]] : []);

//   // console.log(dataInfo.feature_names)

//   return (
//     <div>
//       <div>
//         <h3>Model Statistics</h3>
//         <table className="w-full border-collapse border border-gray-300">
//           <thead>
//             <tr className="bg-gray-100 text-black">
//               <th className="border p-2">Model Name</th>
//               <th className="border p-2">MSE</th>
//               <th className="border p-2">MAE</th>
//               <th className="border p-2">RMSE</th>
//               <th className="border p-2">R²</th>
//               <th className="border p-2">Error</th>
//             </tr>
//           </thead>
//           <tbody>
//             {Object.entries(results.stats).map(([modelName, stats]) => (
//               <tr key={modelName}>
//                 <td className="border p-2">{modelName}</td>
//                 <td className="border p-2">
//                   {typeof stats.mse === "number" ? stats.mse.toFixed(2) : stats.mse ?? "-"}
//                 </td>
//                 <td className="border p-2">
//                   {typeof stats.mae === "number" ? stats.mae.toFixed(2) : stats.mae ?? "-"}
//                 </td>
//                 <td className="border p-2">
//                   {typeof stats.rmse === "number" ? stats.rmse.toFixed(2) : stats.rmse ?? "-"}
//                 </td>
//                 <td className="border p-2">
//                   {typeof stats.r2 === "number"
//                     ? `${(stats.r2 * 100).toFixed(2)}%`
//                     : stats.r2 ?? "-"}
//                 </td>
//                 <td className="border p-2">{stats.error ?? ""}</td>
//               </tr>            
//             ))}
//           </tbody>
//         </table>
//       </div>

//       <div className="flex h-screen">

//         {/* Left panel */}
//         <div className="w-1/4 p-4 border-r border-gray-300">
//           <button
//             onClick={() => setViewMode(viewMode === "predictions" ? "errors" : "predictions")}
//             className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
//           >
//             {viewMode === "predictions" ? "Switch to Errors" : "Switch to Predictions"}
//           </button>

//           <SeriesSelector
//             seriesNames={viewMode === "predictions" ? ["Actual", ...modelNames] : modelNames}
//             selectedSeries={selectedSeries}
//             onChange={setSelectedSeries}
//           />
//         </div>

//         {/* Right panel */}
//         <div className="w-3/4 p-4">
//           <h2 className="text-lg font-bold mt-4">
//             {viewMode === "predictions" ? "Forecasts" : "Prediction Errors"}
//           </h2>
//           <ViewChart
//             chartData={chartData}
//             selectedSeries={selectedSeries}
//           />
//         </div>
//       </div>
//     </div>
//   );
// };

