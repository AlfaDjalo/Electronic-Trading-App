# ViewChart Trade Markers & Position Tracking Enhancement

## Overview
Enhanced `ViewChart` component to support:
1. **Trade Markers** – buy/sell symbols with tooltips showing trade details
2. **Dual Y-Axes** – portfolio value (left) and position size (right)
3. **Position Tracking** – optional rendering of position size over time
4. **Interactive Tooltips** – hover over trades to see execution details

## Backend Changes

### `backtester.py` – Added position tracking
Modified `apply_strategy()` to include `position` in the equity curve array:

```python
equity.append({
    "date": dates[t],
    "value": portfolio_value,    # Portfolio value in $
    "position": position          # Current position size (# shares)
})
```

**Impact**: Every equity point now includes the position size at that timestamp, enabling position charting on a secondary axis.

## Frontend Changes

### 1. Enhanced `ViewChart.jsx`

#### New Props
- `tradeMarkers` (Array): Trade events with structure:
  ```js
  {
    date: "2020-01-01",
    value: 100,         // Trade execution price
    side: "buy",        // "buy" or "sell"
    price: 100,
    shares: 1.2,
    backtestId: "bt1",
    modelName: "Model A"
  }
  ```
- `showPosition` (Boolean): Enable/disable position series on right Y-axis

#### Trade Marker Rendering
- **Buy markers**: Green upward triangles (▲)
- **Sell markers**: Red downward triangles (▼)
- **Interactive tooltips**: Hover to see full trade details
- **ReferenceDot styling**: Color-coded circles at trade locations

#### Dual Y-Axis Setup
```
Left Axis (yAxisId="left"):     Right Axis (yAxisId="right"):
├─ Portfolio value series       ├─ Position series
├─ Actual prices               └─ Hidden when showPosition=false
└─ Trade markers positioned
   on value axis
```

#### Component Structure
```jsx
<ComposedChart>
  <YAxis yAxisId="left" />           // Portfolio value
  {showPosition && <YAxis yAxisId="right" />}  // Position
  
  {selectedSeries.map(name => (
    name.endsWith('_position')
      ? <Line yAxisId="right" />     // Position on right
      : <Line yAxisId="left" />      // Value on left
  ))}
  
  {tradeMarkers.map(marker =>
    <ReferenceDot />                 // Trade markers
  )}
</ComposedChart>
```

### 2. Updated `BacktestResults.jsx`

#### New State
```jsx
const [showTradeMarkers, setShowTradeMarkers] = useState(true);
const [showPosition, setShowPosition] = useState(false);
```

#### Dynamic Series Selection
```jsx
const selectableSeriesNames = useMemo(() => {
  const base = [...seriesNames];
  if (showPosition) {
    // Add position series for each backtest
    return [...base, ...seriesNames.map(s => `${s}_position`)];
  }
  return base;
}, [seriesNames, showPosition]);
```

#### Enhanced Chart Data
- Includes position values alongside portfolio values
- Position keys use naming convention: `{modelName}_position`
- Position data extracted from backend equity arrays

#### UI Controls (above chart)
```jsx
☑ Show trade markers     // Toggle buy/sell symbols
☑ Show position size      // Toggle position on right axis
```

### 3. Trade Marker Data Enrichment

In `BacktestResults`, trade markers now include shares:
```jsx
const tradeMarkers = useMemo(() => {
  const markers = [];
  seriesInfo.forEach(({ id, name, raw }) => {
    const trades = raw?.trades || [];
    trades.forEach((t) => {
      markers.push({
        date: String(date),
        value: price,
        side: side,
        backtestId: id,
        modelName: name,
        shares: t.shares ?? t.size ?? 0  // NEW
      });
    });
  });
  return markers;
}, [seriesInfo]);
```

## UI/UX Flow

1. **User enables "Show position size" checkbox**
   - Backend position data becomes available in chart
   - SeriesSelector shows `{Model}_position` options
   - User can toggle position on/off

2. **User enables "Show trade markers" checkbox**
   - Trade symbols (▲/▼) appear on chart
   - Hover over marker → tooltip with:
     - Trade date
     - Buy/Sell side (color-coded)
     - Execution price
     - Shares executed

3. **Dual-axis scaling**
   - Left axis: auto-scales to portfolio value range
   - Right axis: auto-scales to position range independently
   - No visual conflict due to separate Y-axis domains

## Data Structures

### Equity Array (Backend → Frontend)
```json
{
  "date": "2020-01-01T10:00:00",
  "value": 10500,        // Portfolio value
  "position": 1.2        // Position size in shares
}
```

### Trade Markers (Frontend)
```json
{
  "date": "2020-01-01T10:15:00",
  "value": 10500,
  "side": "buy",
  "price": 100,
  "shares": 1.2,
  "backtestId": "bt1",
  "modelName": "Model A"
}
```

## Styling

### Trade Markers
- Buy (green): `#10b981` (Tailwind green-500)
- Sell (red): `#ef4444` (Tailwind red-500)
- Size: 6px radius circles (ReferenceDot)

### Tooltip
- Background: Dark gray (`bg-gray-900`)
- Border: Subtle white/20 opacity
- Text: Small (`text-xs`) white
- Padding: Compact (`p-2`)

### Axis Labels
- Left: "Portfolio Value"
- Right: "Position Size"
- Rotated 90° for clarity

## Testing

Added test suite in `src/__tests__/ViewChart.test.jsx`:
- ✅ Renders chart with basic data
- ✅ Accepts trade markers prop
- ✅ Renders with showPosition enabled
- ✅ Handles intraday time formatting
- ✅ Shows "No data" message when empty

Run tests:
```bash
npm test
```

## Performance Considerations

1. **Chart re-render**: Triggered only when:
   - `selectedSeries` changes
   - `chartData` changes
   - `tradeMarkers` changes
   - `showPosition` toggles

2. **Data enrichment**: One-pass merge of trades into chart data
   - O(n) where n = number of trade markers
   - No redundant computations

3. **Dual-axis optimization**: Only right Y-axis rendered when needed
   - Conditional rendering: `{showPosition && <YAxis ... />}`

## Backward Compatibility

✅ `ViewChart` works with existing single-axis data
✅ `tradeMarkers` defaults to empty array
✅ `showPosition` defaults to false
✅ Position series optional in chartData

## Future Enhancements

- [ ] Cumulative P&L overlay
- [ ] Max drawdown region highlighting
- [ ] Win/loss rate pie chart in tooltip
- [ ] Trade clustering/grouping for dense periods
- [ ] Position heatmap (color intensity = position size)
- [ ] Custom marker shapes per trade type (limit, market, stop)
