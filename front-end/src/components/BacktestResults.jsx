import React, { useState, useMemo } from "react";
import { SeriesSelector } from "./SeriesSelector";
import { ViewChart } from "./ViewChart";

/**
 * BacktestResults
 *
 * Props:
 *  - backtestResults: object mapping backtestId -> { equity: [{date, value}], trades: [...], stats: {...}, model_name }
 *
 * Renders:
 *  - Top: per-backtest statistics table
 *  - Left: SeriesSelector to pick which equity curves to show
 *  - Right: ViewChart showing equity curves (date/value)
 */
export const BacktestResults = ({ backtestResults }) => {
  const [selectedSeries, setSelectedSeries] = useState([]);
  const [showTradeMarkers, setShowTradeMarkers] = useState(true);
  // Show position on the main chart by default (user requested default true)
  const [showPosition, setShowPosition] = useState(true);
  // If true, position series are drawn on the same (main) chart using the right axis.
  // If false, position series are rendered in a separate chart below the main chart.
  const [positionInline, setPositionInline] = useState(true);

  // Defensive: ensure we have an object
  const results = backtestResults || {};

  // Build list of series names (use model_name if present, else id)
  const seriesInfo = useMemo(() => {
    return Object.entries(results).map(([id, r]) => {
      const name = r?.model_name || String(id);
      return { id, name, raw: r };
    });
  }, [results]);

  const seriesNames = seriesInfo.map((s) => s.name);

  // Build list of selectable series (includes position if showPosition is true)
  const selectableSeriesNames = useMemo(() => {
    const base = [...seriesNames];
    if (showPosition) {
      return [...base, ...seriesNames.map(s => `${s}_position`)];
    }
    return base;
  }, [seriesNames, showPosition]);

  // Default select all series on first render
  React.useEffect(() => {
    if (selectableSeriesNames.length > 0 && selectedSeries.length === 0) {
      setSelectedSeries(selectableSeriesNames.slice());
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectableSeriesNames.length]);

  // Build unified date index (union of all equity dates)
  const chartData = useMemo(() => {
    const dateSet = new Set();
    const seriesMap = {};
    const positionMap = {};

    seriesInfo.forEach(({ id, name, raw }) => {
      const equity = raw?.equity || [];
      seriesMap[name] = {};
      positionMap[`${name}_position`] = {};
      equity.forEach((pt) => {
        // pt expected: { date: ..., value: number, position?: number }
        const d = pt.date || pt.date_time || pt.timestamp || pt["date"];
        const dateStr = String(d);
        dateSet.add(dateStr);
        seriesMap[name][dateStr] = pt.value ?? pt.value ?? pt["value"] ?? null;
        // Track position if it exists in the data
        positionMap[`${name}_position`][dateStr] = pt.position ?? null;
      });
    });

    const allDates = Array.from(dateSet).sort();

    // Map into array of objects suitable for ViewChart
    const out = allDates.map((date) => {
      const obj = { date };
      seriesNames.forEach((sName) => {
        obj[sName] = seriesMap[sName] ? seriesMap[sName][date] ?? null : null;
        // Include position series if showPosition is enabled
        if (showPosition) {
          obj[`${sName}_position`] = positionMap[`${sName}_position`] 
            ? positionMap[`${sName}_position`][date] ?? null 
            : null;
        }
      });
      return obj;
    });

    return out;
  }, [seriesInfo, seriesNames, showPosition]);

  // Build trade markers from trades lists in results
  const tradeMarkers = useMemo(() => {
    const markers = [];

    seriesInfo.forEach(({ id, name, raw }) => {
      const trades = raw?.trades || [];
      trades.forEach((t) => {
        const date = t.date || t.timestamp || t.time;
        const price = t.price ?? t.value ?? t.fill_price ?? null;
        // Attempt to determine buy/sell
        const side = t.side || t.action || t.type || t['buy/sell'] || t['buy/sell:'] || null;
        const shares = t.shares ?? t.size ?? t.quantity ?? 0;
        markers.push({ date: String(date), value: price, side, backtestId: id, modelName: name, shares });
      });
    });

    return markers;
  }, [seriesInfo]);

  if (!backtestResults || Object.keys(backtestResults).length === 0) {
    return <p className="p-4">No backtest results available.</p>;
  }

  return (
    <div>
      <div className="p-4">
        <h3 className="text-xl font-bold mb-4">Backtest Statistics</h3>

        <table className="w-full border-collapse border border-gray-300 mb-6">
          <thead>
            <tr className="bg-gray-100 text-black">
              <th className="border p-2">Backtest</th>
              <th className="border p-2">Model</th>
              <th className="border p-2">Total Return</th>
              <th className="border p-2">Max Drawdown</th>
              <th className="border p-2">Sharpe</th>
              <th className="border p-2">Win Rate</th>
              <th className="border p-2">Trades</th>
            </tr>
          </thead>
          <tbody>
            {seriesInfo.map(({ id, name, raw }) => {
              const stats = raw?.stats || {};
              const trades = Array.isArray(raw?.trades) ? raw.trades.length : (raw?.trades ? 1 : 0);
              return (
                <tr key={id}>
                  <td className="border p-2">{id}</td>
                  <td className="border p-2">{name}</td>
                  <td className="border p-2">{stats?.total_return ?? "-"}</td>
                  <td className="border p-2">{stats?.max_drawdown ?? "-"}</td>
                  <td className="border p-2">{stats?.sharpe ?? "-"}</td>
                  <td className="border p-2">{stats?.win_rate ?? "-"}</td>
                  <td className="border p-2">{trades}</td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <div className="flex h-[65vh]">
          <div className="w-1/4 p-4 border-r border-gray-300">
            <SeriesSelector
              seriesNames={selectableSeriesNames}
              selectedSeries={selectedSeries}
              onChange={setSelectedSeries}
            />
          </div>

          <div className="w-3/4 p-4">
            <div className="mb-3 flex items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <label className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={showTradeMarkers}
                    onChange={(e) => setShowTradeMarkers(e.target.checked)}
                    className="mr-2"
                  />
                  <span className="text-sm">Show trade markers</span>
                </label>

                <label className="inline-flex items-center">
                  <input
                    type="checkbox"
                    checked={showPosition}
                    onChange={(e) => setShowPosition(e.target.checked)}
                    className="mr-2"
                  />
                  <span className="text-sm">Show position size</span>
                </label>

                <label className="inline-flex items-center">
                  <span className="mr-2 text-sm">Position layout:</span>
                  <select
                    value={positionInline ? "inline" : "separate"}
                    onChange={(e) => setPositionInline(e.target.value === "inline")}
                    className="border rounded px-2 py-1 text-sm"
                  >
                    <option value="inline">Inline (right axis)</option>
                    <option value="separate">Separate chart below</option>
                  </select>
                </label>
              </div>
            </div>

            {positionInline ? (
              <ViewChart
                chartData={chartData}
                selectedSeries={selectedSeries}
                tradeMarkers={showTradeMarkers ? tradeMarkers : []}
                showPosition={showPosition}
              />
            ) : (
              <div>
                {/* Main chart: exclude position series */}
                <ViewChart
                  chartData={chartData}
                  selectedSeries={selectedSeries.filter(s => !s.endsWith('_position'))}
                  tradeMarkers={showTradeMarkers ? tradeMarkers : []}
                  showPosition={false}
                />

                {/* Position chart below */}
                <div className="mt-4 h-48 border-t pt-4">
                  {selectedSeries.filter(s => s.endsWith('_position')).length > 0 ? (
                    <ViewChart
                      chartData={chartData}
                      selectedSeries={selectedSeries.filter(s => s.endsWith('_position'))}
                      tradeMarkers={[]}
                      showPosition={false}
                    />
                  ) : (
                    <p className="text-sm text-gray-500">No position series selected to display.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BacktestResults;
