import { useParams, useNavigate } from "react-router-dom";
import { useState, useEffect } from "react";

export const EditModel = ({ modelNames, featureSets, onSave, modelList }) => {
  const { id } = useParams();
  const navigate = useNavigate();

  const model = modelList.find((m) => m.id.toString() === id);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedFeatureSet, setSelectedFeatureSet] = useState("");
  const [normalise, setNormalise] = useState(false);

  useEffect(() => {
    if (model) {
      setSelectedModel(model.model);
      setSelectedFeatureSet(model.featureSet);
      setNormalise(model.normalise);
    }
  }, [model]);

  if (!model) return <p className="p-4">Model not found</p>;

  const handleSave = (e) => {
    e.preventDefault();
    const updated = {
      ...model,
      model: selectedModel,
      featureSet: selectedFeatureSet,
      normalise,
      name: `${selectedModel}-${selectedFeatureSet}`,
    };
    onSave(updated);
    navigate("/models"); // go back to main list
  };

  return (
    <div className="p-6">
      <h2 className="text-xl font-bold mb-4">Edit Model</h2>
      <form onSubmit={handleSave} className="space-y-4">
        {/* Model */}
        <div>
          <label className="block mb-1">Model</label>
          <select
            className="border rounded p-2 w-full text-black bg-white"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            {modelNames.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {/* Feature Set */}
        <div>
          <label className="block mb-1">Feature Set</label>
          <select
            className="border rounded p-2 w-full text-black bg-white"
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

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => navigate("/models")}
            className="bg-gray-300 text-black px-4 py-2 rounded"
          >
            Cancel
          </button>
          <button type="submit" className="bg-blue-600 text-white px-4 py-2 rounded">
            Save Changes
          </button>
        </div>
      </form>
    </div>
  );
}