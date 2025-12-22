import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ComposedChart,
  ReferenceDot
} from "recharts";
import { useState } from "react";

/**
 * CustomTradeMarker
 * 
 * Renders buy (green ▲) and sell (red ▼) markers at trade locations.
 */
const CustomTradeMarker = ({ x, y, payload, side }) => {
  if (typeof x !== 'number' || typeof y !== 'number') return null;
  
  const isGreen = side?.toLowerCase() === 'buy';
  const color = isGreen ? '#10b981' : '#ef4444'; // tailwind green/red
  const symbol = isGreen ? '▲' : '▼';

  return (
    <g>
      <text
        x={x}
        y={y - 10}
        textAnchor="middle"
        fill={color}
        fontSize={14}
        fontWeight="bold"
      >
        {symbol}
      </text>
    </g>
  );
};

/**
 * CustomTooltip for trade markers
 * 
 * Displays trade details: date, side, price, shares
 */
const TradeTooltip = ({ active, payload }) => {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data?.tradeInfo) return null;

  const { date, side, price, shares } = data.tradeInfo;

  return (
    <div className="bg-gray-900 border border-white/20 rounded p-2 text-xs text-white">
      <p><strong>Trade</strong></p>
      <p>Date: {date}</p>
      <p>Side: <span className={side?.toLowerCase() === 'buy' ? 'text-green-400' : 'text-red-400'}>{side}</span></p>
      <p>Price: ${parseFloat(price).toFixed(2)}</p>
      <p>Shares: {parseFloat(shares).toFixed(2)}</p>
    </div>
  );
};

export const ViewChart = ({ chartData, selectedSeries, tradeMarkers = [], showPosition = false }) => {

    console.log("chartData sample:", chartData[0]);
    console.log("chartData sample:", chartData[1]);
    console.log("chartData sample:", chartData[2]);
    console.log("selectedSeries:", selectedSeries);
    console.log("tradeMarkers:", tradeMarkers);

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

    // Merge trade markers into chartData for rendering on same timeline
    const enrichedChartData = chartData.map(point => {
        const tradeAtDate = tradeMarkers.find(m => m.date === point.date);
        if (tradeAtDate) {
            return {
                ...point,
                tradeInfo: {
                    date: tradeAtDate.date,
                    side: tradeAtDate.side,
                    price: tradeAtDate.value,
                    shares: tradeAtDate.shares || 0
                }
            };
        }
        return point;
    });

    return (
        // <div className="w-full h-[600px]" style={{border: '2px solid red', backgroundColor: 'lightgray'}}>
        //     <ResponsiveContainer width="100%" height="100%"></ResponsiveContainer>        
        <div className="w-full h-[600px]">
            <ResponsiveContainer width="100%" height="100%">
                <ComposedChart 
                    data={enrichedChartData}
                    key={selectedSeries.join(',')}
                    margin={{ top: 20, right: 80, bottom: 20, left: 80 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                        dataKey="date"
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

                    {/* Left Y-axis for portfolio value */}
                    <YAxis
                        yAxisId="left"
                        tick={{ fontSize: 12 }}
                        width={80}
                        domain={["auto", "auto"]}
                        label={{ value: 'Portfolio Value', angle: -90, position: 'insideLeftTop', offset: 10 }}
                    />

                    {/* Right Y-axis for position (only if showPosition is true) */}
                    {showPosition && (
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            tick={{ fontSize: 12 }}
                            width={80}
                            domain={["auto", "auto"]}
                            label={{ value: 'Position Size', angle: 90, position: 'insideRightTop', offset: 10 }}
                        />
                    )}

                    {/* Combined tooltip for lines and markers */}
                    <Tooltip content={<TradeTooltip />} />
                    <Legend />

                    {/* Render selected series lines on left axis */}
                    {selectedSeries.map((name, idx) => {
                        // Determine which axis to use
                        let yAxisId = "left";
                        let strokeColor = `hsl(${(idx * 70) % 360}, 70%, 50%)`;

                        // Position series goes on right axis if show position is enabled
                        if (showPosition && name.endsWith('_position')) {
                            yAxisId = "right";
                            strokeColor = '#3b82f6'; // blue for position
                        }

                        return (
                            <Line
                                key={name}
                                yAxisId={yAxisId}
                                type="monotone"
                                dataKey={name}
                                strokeWidth={2}
                                dot={false}
                                stroke={strokeColor}
                                isAnimationActive={false}
                            />
                        );
                    })}

                    {/* Render trade markers (buy/sell points with custom symbols) */}
                    {tradeMarkers.length > 0 && enrichedChartData.map((point, idx) => {
                        if (!point.tradeInfo) return null;
                        const { side, price } = point.tradeInfo;
                        
                        return (
                            <ReferenceDot
                                key={`trade-${idx}`}
                                x={point.date}
                                y={price}
                                yAxisId="left"
                                r={6}
                                fill={side?.toLowerCase() === 'buy' ? '#10b981' : '#ef4444'}
                                onClick={() => {
                                    // Optional: add click handlers for trade details
                                    console.log("Trade marker clicked:", point.tradeInfo);
                                }}
                            >
                                <TradeTooltip />
                            </ReferenceDot>
                        );
                    })}
                </ComposedChart>
            </ResponsiveContainer>
        </div>
    );
};
