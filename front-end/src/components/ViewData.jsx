import { useState } from "react";
import { FeatureSelector } from "./FeatureSelector";
import { DataSummary } from "./DataSummary";
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid } from "recharts";

export const ViewData = ({ chartData, featureNames }) => {
  const [selectedFeatures, setSelectedFeatures] = useState([]);

  console.log("Chart Data:", chartData);
  console.log("Selected Features:", featureNames);

  if (!chartData || chartData.length === 0) {
    return <p className="p-4">No data loaded yet.</p>;
  }

  return (
    <div>
        <div className="flex h-screen">
            {/* <DataSummary /> */}

            {/* Left panel */}
            <div className="w-1/4 p-4 border-r border-gray-300">
                <FeatureSelector
                features={featureNames}
                selectedFeatures={selectedFeatures}
                onChange={setSelectedFeatures}
                />
            </div>

            {/* Right panel */}
            <div className="w-3/4 p-4">
                <LineChart width={800} height={500} data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="Date" />
                <YAxis domain={['auto', 'auto']} />
                <Tooltip />
                <Legend />
                {selectedFeatures.map((feature, idx) => (
                    <Line
                    key={feature}
                    type="monotone"
                    dataKey={feature}
                    stroke={`hsl(${idx * 50}, 70%, 50%)`} // different color for each
                    dot={false}
                    />
                ))}
                </LineChart>
            </div>
        </div>
    </div>
  );
};