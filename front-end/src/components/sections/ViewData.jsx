import { useState, useEffect } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, Legend, CartesianGrid, ResponsiveContainer
} from "recharts";

export default function ViewData({ ticker }) {
  const [features, setFeatures] = useState([]); 
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [data, setData] = useState([]);
  const [axisAssignment, setAxisAssignment] = useState({}); // feature -> axis

  // Load features + data
  useEffect(() => {
    fetch(`/api/features?ticker=${ticker}`)
      .then((res) => res.json())
      .then((json) => {
        setFeatures(json.features);
        setSelectedFeatures([json.features[0]]);
        setData(transformData(json.dates, json.featuresData));
      });
  }, [ticker]);

  const transformData = (dates, featuresData) =>
    dates.map((date, i) => {
      const point = { date };
      for (let f in featuresData) {
        point[f] = featuresData[f][i];
      }
      return point;
    });

  const toggleFeature = (feature) => {
    setSelectedFeatures((prev) =>
      prev.includes(feature)
        ? prev.filter((f) => f !== feature)
        : [...prev, feature]
    );
  };

  // Assign default axes when features change
  useEffect(() => {
    if (selectedFeatures.length <= 1 || data.length === 0) {
      setAxisAssignment({});
      return;
    }

    // Compute ranges
    const ranges = {};
    selectedFeatures.forEach((f) => {
      const vals = data.map((d) => d[f]).filter((v) => v != null);
      ranges[f] = Math.max(...vals) - Math.min(...vals);
    });

    // Biggest range -> right axis, others -> left (only if not manually set yet)
    const maxFeature = Object.entries(ranges).sort((a, b) => b[1] - a[1])[0][0];
    const newAssignment = { ...axisAssignment };
    selectedFeatures.forEach((f) => {
      if (!newAssignment[f]) {
        newAssignment[f] = f === maxFeature ? "right" : "left";
      }
    });
    setAxisAssignment(newAssignment);
  }, [selectedFeatures, data]);

  // Toggle axis manually
  const toggleAxis = (feature) => {
    setAxisAssignment((prev) => ({
      ...prev,
      [feature]: prev[feature] === "left" ? "right" : "left",
    }));
  };

  return (
    <div className="container mt-4">
      <h3>View feature charts for {ticker}</h3>

      <div className="mb-3">
        {features.map((f) => (
          <div key={f} className="d-inline-block me-4">
            <label>
              <input
                type="checkbox"
                checked={selectedFeatures.includes(f)}
                onChange={() => toggleFeature(f)}
              />{" "}
              {f}
            </label>
            {selectedFeatures.includes(f) && (
              <button
                className="btn btn-sm btn-outline-secondary ms-2"
                onClick={() => toggleAxis(f)}
              >
                Axis: {axisAssignment[f] || "left"}
              </button>
            )}
          </div>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={500}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" orientation="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />

          {selectedFeatures.map((f, i) => (
            <Line
              key={f}
              type="monotone"
              dataKey={f}
              stroke={`hsl(${(i * 60) % 360}, 70%, 50%)`}
              dot={false}
              yAxisId={axisAssignment[f] || "left"}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
