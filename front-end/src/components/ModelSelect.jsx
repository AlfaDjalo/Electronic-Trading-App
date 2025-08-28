import { useState } from "react";

import { ModelForm } from "./ModelForm";
// import { ModelAdd } from "./ModelAdd";

export const ModelSelect = ({
  modelNames,
  featureSets,
  modelList,
  onAddModel,
  onEdit,
  onDelete,
  onSetParameters,
}) => {
  const [mode, setMode] = useState("add");
  const [editingModel, setEditingModel] = useState(null);

  const handleAdd = (newModel) => {
    onAddModel(newModel);
    setMode("add");
    setEditingModel(null);
  };

  const handleUpdate = (updatedModel) => {
    onEdit(updatedModel);
    setMode("add");
    setEditingModel(null);
  };

//   const [selectedModel, setSelectedModel] = useState(modelNames?.[0] || "");
//   const [selectedFeatureSet, setSelectedFeatureSet] = useState(featureSets?.[0] || "");
//   const [normalise, setNormalise] = useState(false);

//   const handleAddModel = (e) => {
//     e.preventDefault();
//     const newModel = {
//       id: Date.now(), // parent could override with backend id
//       name: `${selectedModel}-${selectedFeatureSet}`,
//       model: selectedModel,
//       featureSet: selectedFeatureSet,
//       normalise,
//     };
//     onAddModel?.(newModel);

//     // reset normalise if you want
//     setNormalise(false);
//   };

  return (
    <div className="p-4">
      <h3 className="text-xl font-bold mb-4">
        {mode === "add" ? "Add New Model" : "Edit Model"}
      </h3>      
      
      {/* <ModelAdd onAddModel={onAddModel} modelNames={modelNames} featureSets={featureSets} /> */}
      <ModelForm
        modelNames={modelNames}
        featureSets={featureSets}
        initialValues={editingModel}
        onSubmit={mode === "add" ? handleAdd : handleUpdate}
      />

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
                    className="bg-blue-500 text-white px-2 py-1 rounded"
                    onClick={() => {
                      setEditingModel(m);
                      setMode("edit");
                    }}
                  >
                    Edit
                  </button>
                  {/* <button
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
                  </button> */}
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