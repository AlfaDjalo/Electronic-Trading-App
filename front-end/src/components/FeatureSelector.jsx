import Select from "react-select";

export const FeatureSelector = ({ features, selectedFeatures, onChange }) => {
  const options = features.map(f => ({ value: f, label: f }));

  return (
    <div className="p-4 bg-gray-100 h-screen overflow-y-auto">
      <h2 className="text-lg font-bold mb-2">Select Features</h2>

        <Select
            isMulti
            name="features"
            options={options}
            value={options.filter(o => selectedFeatures.includes(o.value))}
            onChange={(selected) => onChange(selected.map(s => s.value))}
            placeholder="Select features to plot..."
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

