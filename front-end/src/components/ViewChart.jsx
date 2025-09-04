import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer
} from "recharts";

export const ViewChart = ({ chartData, selectedSeries }) => {

    // console.log("chartData sample:", chartData[0]);
    // console.log("chartData sample:", chartData[1]);
    // console.log("chartData sample:", chartData[2]);
    // console.log("selectedSeries:", selectedSeries);

    if (!chartData || chartData.length === 0) {
        return <p className="p-4">No chart data available.</p>;
    }

    const isIntraday = (chartData) => {
    if (!chartData || chartData.length < 2) return false;
    const t0 = new Date(chartData[0].date);
    const t1 = new Date(chartData[1].date);
    return (t1 - t0) < 24 * 60 * 60 * 1000; // < 1 day
    };

    const intraday = isIntraday(chartData);

    // Add this debugging:
    // console.log("Keys in chartData[0]:", Object.keys(chartData[0] || {}));
    // selectedSeries.forEach(series => {
    //     const hasData = chartData.some(point => point[series] !== undefined);
    //     console.log(`Series "${series}" exists in data:`, hasData);
    //     if (hasData) {
    //         const values = chartData.slice(0, 5).map(point => point[series]);
    //         console.log(`First 5 values for "${series}":`, values);
    //     }
    // });

    return (
        // <div className="w-full h-[600px]" style={{border: '2px solid red', backgroundColor: 'lightgray'}}>
        //     <ResponsiveContainer width="100%" height="100%"></ResponsiveContainer>        
        <div className="w-full h-[600px]">
            <ResponsiveContainer width="100%" height="100%">
                <LineChart 
                    data={chartData}
                    key={selectedSeries.join(',')}
                    margin={{ top: 20, right: 20, bottom: 20, left: 80 }}
                >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                    dataKey="date"
                    // domain={['auto', 'auto']}
                    tick={{ fontSize: intraday ? 10 : 12 }}
                    minTickGap={intraday ? 5 : 20}
                    tickFormatter={(tick) => {
                        const d = new Date(tick);
                        if (intraday) {
                        return `${d.getHours()}:${d.getMinutes()}:${d.getSeconds()}`;
                        } else {
                        return `${d.getFullYear()}-${d.getMonth()+1}-${d.getDate()}`;
                        }
                    }}
                />            

                <YAxis
                    tick={{ fontSize: 12 }}
                    width={80}
                    domain={["auto", "auto"]} // ✅ auto axis scale
                />
                <Tooltip />
                <Legend />

                {selectedSeries.map((name, idx) => (
                    <Line
                    key={name}
                    type="monotone"
                    dataKey={name}
                    strokeWidth={2}
                    dot={false}
                    stroke={ `hsl(${(idx * 70) % 360}, 70%, 50%)` }
                    // stroke={
                    //     idx === 0
                    //     ? "#ffffff" // Actual in white
                    //     : `hsl(${(idx * 70) % 360}, 70%, 50%)` // auto colors for models
                    // }
                    />
                ))}
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
};
