import { useState } from "react";
import { SeriesSelector } from "./SeriesSelector";
import { ViewChart } from "./ViewChart";
// import { DataSummary } from "./DataSummary";

export const ViewData = ({ data, seriesNames }) => {
  const [selectedSeries, setSelectedSeries] = useState(seriesNames.length > 1 ? [seriesNames[0]] : []);

  if (!data || data.length === 0) {
    return <p className="p-4">No data loaded yet.</p>
  }

  return (
    <div>
      <div className="flex h-screen">

        {/* Left panel */}
        <div className="w-1/4 p-4 border-r border-gray-300">
          <SeriesSelector
            seriesNames={seriesNames}
            selectedSeries={selectedSeries}
            onChange={setSelectedSeries}
          />
        </div>

        {/* Right panel */}
        <div className="w-3/4 p-4">
          <ViewChart chartData={data} selectedSeries={selectedSeries} />
        </div>
      </div>
      {/* <DataSummary data={data} /> */}
    </div>
  );
};

