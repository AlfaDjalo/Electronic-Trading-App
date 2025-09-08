import { useState, useEffect } from "react";

export const FeatureForm = ({   
    seriesNames = [],
    functionList = [],
    initialValues = {}, 
    onSubmit
}) => {
  const safeInitial = initialValues || {};

  const [name, setName] = useState(safeInitial.name || "");
  const [inputFields, setInputFields] = useState(safeInitial.input_data_fields || []);
  const [func, setFunc] = useState(safeInitial.function || "");
  const [params, setParams] = useState(
    safeInitial.function_parameters ? JSON.stringify(safeInitial.function_parameters) : "{}"
  );

  useEffect(() => {
    const iv = initialValues || {};
    setName(iv.name || "");
    setInputFields(iv.input_data_fields || []);
    setFunc(iv.function || "");
    setParams(iv.function_parameters ? JSON.stringify(iv.function_parameters) : "{}");
  }, [initialValues]);

  const handleSubmit = (e) => {
    e.preventDefault();

    let parsedParams = {};
    try {
      parsedParams = JSON.parse(params || "{}");
    } catch {
      alert("Parameters must be valid JSON");
      return;
    }

    onSubmit({
      ...safeInitial,
      id: safeInitial.id ?? Date.now(),
      name,
      input_data_fields: inputFields,
      function: func,
      function_parameters: parsedParams,
    });

    // Reset only in Add mode
    if (!safeInitial.id) {
      setName("");
      setInputFields("");
      setFunc("");
      setParams("{}");
    }
  };

  return (
    <form
        onSubmit={handleSubmit}
        className="bg-white p-4 rounded-lg shadow-md mb-6 grid grid-cols-1 md:grid-cols-2 gap-4"
    >
    {/* Top left: Name */}
        <div>
            <label className="block mb-1 text-sm font-medium text-gray-700">Name</label>
            <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full border border-gray-300 rounded px-3 py-2 text-black"
            disabled={name === "Target"} // optional safeguard
            />
        </div>

        {/* Top right: Function */}
        <div>
            <label className="block mb-1 text-sm font-medium text-gray-700">Function</label>
            <select
            value={func}
            onChange={(e) => setFunc(e.target.value)}
            className="w-full border px-3 py-2 rounded text-black"
            >
            <option value="">Select function...</option>
            {functionList.map((fn) => (
                <option key={fn} value={fn}>
                {fn}
                </option>
            ))}
            </select>
        </div>

        {/* Bottom left: Input Fields */}
        <div>
            <label className="block mb-1 text-sm font-medium text-gray-700">Input Fields</label>
            <select
            multiple
            value={inputFields}
            onChange={(e) =>
                setInputFields(Array.from(e.target.selectedOptions, (o) => o.value))
            }
            className="w-full border px-3 py-2 rounded text-black"
            >
            {seriesNames.map((s) => (
                <option key={s} value={s}>
                {s}
                </option>
            ))}
            </select>
        </div>

        {/* Bottom right: Parameters */}
        <div>
            <label className="block mb-1 text-sm font-medium text-gray-700">Parameters (JSON)</label>
            <textarea
            rows={3}
            value={params}
            onChange={(e) => setParams(e.target.value)}
            className="w-full border border-gray-300 rounded px-3 py-2 text-black font-mono text-sm"
            />
        </div>

        {/* Submit button spans both columns */}
        <div className="col-span-1 md:col-span-2 flex justify-center pt-2">
            <button
            type="submit"
            className="px-6 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
            >
            {initialValues?.id ? "Update Feature" : "Add Feature"}
            </button>
        </div>
    </form>
  );
};
