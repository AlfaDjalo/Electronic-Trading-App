import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";

export const ViewResults = ({ results }) => {
  const { dates, actual, predictions } = results;

  // console.log(dates)
  // console.log(actual)
  // console.log(predictions)

  // Rebuild data into recharts-friendly array of objects
  const chartData = dates.map((date, i) => {
    const row = { date, Actual: actual[i] };
    for (const [model, preds] of Object.entries(predictions)) {
      row[model] = preds[i];
    }
    return row;
  });

  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-4">Predictions vs Actual</h2>
      <LineChart width={1000} height={400} data={chartData}>
        <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
        <XAxis dataKey="date" tick={{ fontSize: 12 }} />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="Actual" stroke="#000" strokeDasharray="5 5" />
        {Object.keys(predictions).map((model, idx) => (
          <Line
            key={model}
            type="monotone"
            dataKey={model}
            stroke={`hsl(${(idx * 60) % 360}, 70%, 50%)`}
          />
        ))}
      </LineChart>
    </div>
  );
};