import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ModelForm } from "./ModelForm";
import { ModelRun } from "./ModelRun"; 

export const ModelSelect = ({
  modelConfig,
  featureSets,
  modelList,
  onAddModel,
  onEdit,
  onDelete,
  rawData,
  setResults
}) => {
  const [editingModel, setEditingModel] = useState(null);

  const navigate = useNavigate();

  const handleAdd = (newModel) => {
    const modelWithId = { ...newModel, id: newModel.id ?? Date.now() };
    onAddModel(modelWithId);
    setEditingModel(null);
  };

  const handleUpdate = (updatedModel) => {
    onEdit(updatedModel);
    setEditingModel(null);
  };

  return (
    <div className="p-4">
      {/* <h3 className="text-xl font-bold mb-4">
        {mode === "add" ? "Add New Model" : "Edit Model"}
      </h3>       */}
      
      {/* <ModelAdd onAddModel={onAddModel} modelNames={modelNames} featureSets={featureSets} /> */}
      <ModelForm
        modelConfig={modelConfig}
        featureSets={Object.keys(featureSets || {})}
        initialValues={editingModel}
        onSubmit={editingModel ? handleUpdate : handleAdd}
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
              <th className="border p-2">Forecast Period</th>
              <th className="border p-2">Input Width</th>
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
                <td className="border p-2">{m.forecastPeriod}</td>
                <td className="border p-2">{m.inputWidth}</td>
                <td className="border p-2">{m.normalise ? "Yes" : "No"}</td>
                <td className="border p-2 space-x-2">
                  <button
                    className="bg-blue-500 text-white px-2 py-1 rounded"
                    onClick={() => {
                      setEditingModel(m);
                    }}
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

      {modelList.length > 0 && (
        <div className="flex justify-center mt-6">
          <button
            className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
            onClick={async () => {
              try {
                // Pass all models to ModelRun
                const results = await ModelRun(modelList, rawData); 
                // console.log("Run results:", results);
                setResults(results);            // save in App.js state
                navigate("/view_results");   // go to results page
                // TODO: handle displaying results in your UI
              } catch (err) {
                console.error("Error running models:", err);
              }
            }}
          >
            Run Models
          </button>
        </div>
      )}
    </div>
  );
};