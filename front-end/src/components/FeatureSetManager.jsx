import React, { useState, useEffect } from 'react';
import { FeatureForm } from "./FeatureForm";

export const FeatureSetManager = ({ availableFeatureSets, seriesNames, functionList }) => {
  const [selectedFeatureSet, setSelectedFeatureSet] = useState('');
  const [currentFeatureSet, setCurrentFeatureSet] = useState(null);
  const [editingFeature, setEditingFeature] = useState(null);

  useEffect(() => {
    const firstKey = Object.keys(availableFeatureSets || {})[0];
    if (firstKey) {
      setSelectedFeatureSet(firstKey);
      setCurrentFeatureSet(availableFeatureSets[firstKey]);
    }
  }, [availableFeatureSets]);

  const handleFeatureSetChange = (featureSetName) => {
    setSelectedFeatureSet(featureSetName);
    setCurrentFeatureSet(availableFeatureSets[featureSetName] || null);
  };

  const handleAdd = (newFeature) => {
    setCurrentFeatureSet((prev) => {
      if (prev.features.some((f) => f.name === newFeature.name)) {
        alert(`Feature with name "${newFeature.name}" already exists!`);
        return prev; // don’t add duplicate
      }
      return {
        ...prev,
        features: [...(prev?.features || []), newFeature],
      };
    });
    setEditingFeature(null);
  };

  const handleUpdate = (updatedFeature) => {
    setCurrentFeatureSet((prev) => {
      if (
        prev.features.some(
          (f) => f.name === updatedFeature.name && f.id !== updatedFeature.id
        )
      ) {
        alert(`Another feature with name "${updatedFeature.name}" already exists!`);
        return prev; // don’t allow duplicate rename
      }
      return {
        ...prev,
        features: prev.features.map((f) =>
          f.id === updatedFeature.id ? updatedFeature : f
        ),
      };
    });
    setEditingFeature(null);
  };

  const handleDelete = (id) => {
    setCurrentFeatureSet((prev) => ({
      ...prev,
      features: prev.features.filter((f) => f.id !== id),
    }));
  };

  // ✅ Compute orderedFeatures outside JSX
  const orderedFeatures = currentFeatureSet
    ? [
        ...(currentFeatureSet.features || []).filter(
          (f) => f.name?.toLowerCase() !== "target"
        ),
        ...(currentFeatureSet.features || []).filter(
          (f) => f.name?.toLowerCase() === "target"
        ),
      ]
    : [];

  const handleSave = async () => {
    if (!selectedFeatureSet || !currentFeatureSet) return;

    try {
      const res = await fetch("http://localhost:5000/api/save_feature_set", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: selectedFeatureSet,
          feature_set: currentFeatureSet,
        }),
      });

      const result = await res.json();
      if (res.ok) {
        alert(`✅ ${result.message}`);
      } else {
        alert(`❌ Error: ${result.error}`);
      }
    } catch (err) {
      alert(`❌ Save failed: ${err.message}`);
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Feature Set Manager</h1>

      {/* Feature Set Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Select Feature Set:</label>
        <select
          value={selectedFeatureSet}
          onChange={(e) => handleFeatureSetChange(e.target.value)}
          className="min-w-[200px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select a feature set...</option>
          {Object.keys(availableFeatureSets).map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {/* Features Table */}
      {currentFeatureSet && (
        <div className="p-4">
          <div className="mb-4 text-white">
            <p>
              <span className="font-semibold">Frequency:</span>
              {currentFeatureSet.data_type || "N/A"}
            </p>
          </div>

          <div className="p-4 border-b">
            <h2 className="text-xl font-semibold">Features</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse border border-gray-300">
              <thead className="bg-gray-100 text-black">
                <tr>
                  <th className="border p-2">Name</th>
                  <th className="border p-2">Input Fields</th>
                  <th className="border p-2">Function</th>
                  <th className="border p-2">Parameters</th>
                  <th className="border p-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {orderedFeatures.map((f, idx) => (
                  <tr
                    key={f.id ?? `feature-${idx}`}
                    className={f.name?.toLowerCase() === "target" ? "font-semibold" : ""}
                  >
                    <td className="border p-2">{f.name}</td>
                    <td className="border p-2">{f.input_data_fields.join(", ")}</td>
                    <td className="border p-2">{f.function}</td>
                    <td className="border p-2">{JSON.stringify(f.function_parameters)}</td>
                    <td className="border p-2 space-x-2">
                      <button
                        className="bg-blue-500 text-white px-2 py-1 rounded"
                        onClick={() => setEditingFeature(f)}
                      >
                        Edit
                      </button>
                      {f.name.toLowerCase() !== "target" && (
                        <button
                          className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
                          onClick={() => handleDelete(f.id)}
                        >
                          Delete
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-4 flex justify-end">
              <button
                onClick={handleSave}
                className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
              >
                Save Feature Set
              </button>
            </div>            
          </div>
        </div>
      )}

      <FeatureForm
        seriesNames={seriesNames}
        functionList={functionList}
        initialValues={editingFeature}
        existingNames={(currentFeatureSet?.features || []).map(f => f.name)}
        onSubmit={editingFeature ? handleUpdate : handleAdd}
      />
    </div>
  );
};
