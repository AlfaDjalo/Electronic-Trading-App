import Select from "react-select";
import { useEffect } from "react";

export const SeriesSelector = ({ seriesNames, selectedSeries, onChange }) => {

  const options = seriesNames.map(f => ({ value: f, label: f }));

    // Ensure at least one feature is always selected
  useEffect(() => {
    if (selectedSeries.length === 0 && options.length > 0) {
      onChange([options[0].value]); // auto-select first feature
    }
  }, [selectedSeries, options, onChange]);

  return (
    <div className="p-4 bg-gray-100 h-screen overflow-y-auto">
      <h2 className="text-lg font-bold mb-2">Select Series</h2>

        <Select
            isMulti
            name="series"
            options={options}
            value={options.filter(o => selectedSeries.includes(o.value))}
            onChange={(selected) => onChange(selected.map(s => s.value))}
            placeholder="Select seres to plot..."
            className="basic-multi-select"
            classNamePrefix="select"
            styles={{
            control: (provided) => ({
                ...provided,
                backgroundColor: "white",
                color: "black",
            }),
            menu: (provided) => ({
                ...provided,
                backgroundColor: "white",
                color: "black",
            }),
            multiValueLabel: (provided) => ({
                ...provided,
                color: "black",
            }),
            }}
        />
    </div>
  );
};

