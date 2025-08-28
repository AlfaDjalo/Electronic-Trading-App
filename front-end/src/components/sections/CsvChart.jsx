// import { useState } from "react";
// import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from "recharts";

// export const CsvChart = ({ data, columns }) => {
//   const [selectedFeature, setSelectedFeature] = useState(columns[0]); // default to first column

//   if (!data || data.length === 0) {
//     return <p>No data loaded yet</p>;
//   }

//   return (
//     <div className="p-4">
//       {/* Dropdown to pick feature */}
//       <select
//         value={selectedFeature}
//         onChange={(e) => setSelectedFeature(e.target.value)}
//         className="p-2 border rounded"
//       >
//         {columns.map((col) => (
//           <option key={col} value={col}>
//             {col}
//           </option>
//         ))}
//       </select>

//       {/* Chart */}
//       <LineChart
//         width={800}
//         height={400}
//         data={data}
//         margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
//       >
//         <CartesianGrid strokeDasharray="3 3" />
//         <XAxis dataKey="Date" />  {/* adjust if your x-axis column is different */}
//         <YAxis />
//         <Tooltip />
//         <Legend />
//         <Line type="monotone" dataKey={selectedFeature} stroke="#8884d8" />
//       </LineChart>
//     </div>
//   );
// };
