export const ModelParametersForm = ({ config, values, onChange }) => {
  return (
    <form className="space-y-4">
      {Object.entries(config).map(([param, def]) => (
        <div key={param}>
          <label className="block font-medium mb-1">{param}</label>

          {def.type === "integer" || def.type === "float" ? (
            <input
              type="number"
              min={def.min}
              max={def.max}
              step={def.type === "float" ? "0.01" : "1"}
              value={values[param] ?? def.default}
              onChange={(e) => onChange(param, e.target.value)}
              className="border rounded p-2 w-full text-black"
            />
          ) : def.type === "category" ? (
            <select
              value={values[param] ?? def.default}
              onChange={(e) => onChange(param, e.target.value)}
              className="border rounded p-2 w-full text-white bg-black"
            >
              {def.values.map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          ) : null}
        </div>
      ))}
    </form>
  );
};
