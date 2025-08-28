import { useState } from "react";
import { Link } from "react-router-dom";

export const SelectModel = ({
  modelNames,
  featureSets,
  modelList,
  onAddModel,
  onEdit,
  onDelete,
  onSetParameters,
}) => {
  const [selectedModel, setSelectedModel] = useState(modelNames?.[0] || "");
  const [selectedFeatureSet, setSelectedFeatureSet] = useState(featureSets?.[0] || "");
  const [normalise, setNormalise] = useState(false);

  const handleAddModel = (e) => {
    e.preventDefault();
    const newModel = {
      id: Date.now(), // parent could override with backend id
      name: `${selectedModel}-${selectedFeatureSet}`,
      model: selectedModel,
      featureSet: selectedFeatureSet,
      normalise,
    };
    onAddModel?.(newModel);

    // reset normalise if you want
    setNormalise(false);
  };

  return (
    <div className="p-4">
      <h3 className="text-xl font-bold mb-4">Add New Model</h3>

      <form onSubmit={handleAddModel} className="space-y-4">
        {/* Select Model */}
        <div>
          <label className="block mb-1">Model</label>
          <select
            className="border rounded p-2 w-auto min-w-[12rem] max-w-xs text-black bg-white"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modelNames.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
        </div>

        {/* Select Feature Set */}
        <div>
          <label className="block mb-1">Feature Set</label>
          <select
            className="border rounded p-2 w-auto min-w-[12rem] max-w-xs text-black bg-white"
            value={selectedFeatureSet}
            onChange={(e) => setSelectedFeatureSet(e.target.value)}
          >
            {featureSets.map((fs) => (
              <option key={fs} value={fs}>
                {fs}
              </option>
            ))}
          </select>
        </div>

        {/* Normalise */}
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="normalise"
            checked={normalise}
            onChange={(e) => setNormalise(e.target.checked)}
          />
          <label htmlFor="normalise">Normalise</label>
        </div>

        <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded">
          Add Model
        </button>
      </form>

      <hr className="my-6" />

      <h4 className="text-lg font-semibold mb-3">Existing Models</h4>
      {modelList.length === 0 ? (
        <p className="text-gray-500">No models added yet.</p>
      ) : (
        <table className="w-full border-collapse border border-gray-300">
          <thead>
            <tr className="bg-gray-100 text-black">
              <th className="border p-2">Name</th>
              <th className="border p-2">Model</th>
              <th className="border p-2">Feature Set</th>
              <th className="border p-2">Normalise</th>
              <th className="border p-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {modelList.map((m) => (
              <tr key={m.id}>
                <td className="border p-2">{m.name}</td>
                <td className="border p-2">{m.model}</td>
                <td className="border p-2">{m.featureSet}</td>
                <td className="border p-2">{m.normalise ? "Yes" : "No"}</td>
                <td className="border p-2 space-x-2">
                  <button
                    className="bg-blue-700 text-white px-2 py-1 rounded border border-blue-900"
                    onClick={() => onSetParameters?.(m)}
                  >
                    Set Parameters
                  </button>
                  <button
                    className="bg-blue-500 text-white px-2 py-1 rounded border border-blue-700"
                    onClick={() => onEdit?.(m)}
                  >
                    Edit
                  </button>
                  <button
                    className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
                    onClick={() => onDelete?.(m.id)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};
