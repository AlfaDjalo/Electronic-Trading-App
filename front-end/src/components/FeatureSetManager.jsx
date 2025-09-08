import React, { useState, useEffect } from 'react';
import { FeatureForm } from "./FeatureForm";

export const FeatureSetManager = ({ availableFeatureSets, seriesNames, functionList }) => {
  const [selectedFeatureSet, setSelectedFeatureSet] = useState('');
  const [currentFeatureSet, setCurrentFeatureSet] = useState(null);
  const [editingFeature, setEditingFeature] = useState(null);

  // const functionList = ["sma", "ema", "bollinger_bands", "momentum"]

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
    setCurrentFeatureSet((prev) => ({
      ...prev,
      features: [...(prev?.features || []), newFeature],
    }));
    setEditingFeature(null);
  };

  const handleUpdate = (updatedFeature) => {
    setCurrentFeatureSet((prev) => ({
      ...prev,
      features: prev.features.map((f) =>
        f.id === updatedFeature.id ? updatedFeature : f
      ),
    }));
    setEditingFeature(null);
  };

  const handleDelete = (id) => {
    setCurrentFeatureSet((prev) => ({
      ...prev,
      features: prev.features.filter((f) => f.id !== id),
    }));
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Feature Set Manager</h1>

      <FeatureForm
        seriesNames={seriesNames}
        functionList={functionList}
        initialValues={editingFeature}
        onSubmit={editingFeature ? handleUpdate : handleAdd}
      />

      {/* Feature Set Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2">Select Feature Set:</label>
        <select 
          value={selectedFeatureSet} 
          onChange={(e) => handleFeatureSetChange(e.target.value)}
          className="min-w-[200px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
          // className="px-3 py-2 border border-gray-300 rounded-md"
        >
          <option value="">Select a feature set...</option>
          {Object.keys(availableFeatureSets).map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </div>

      {/* Features Table */}
      {currentFeatureSet && (
        <div className="p-4">
          {/* Meta Info */}
          <div className="mb-4 text-white">
            <p>
              <span className="font-semibold">Frequency:</span>
              {currentFeatureSet.data_type || "N/A"}
            </p>
          </div>
        {/* <div className="bg-white border border-gray-200 rounded-lg"> */}
          <div className="p-4 border-b">
            <h2 className="text-xl font-semibold">Features</h2>
          </div>
          
          <div className="overflow-x-auto">

            {/* Existing features table */}
            {currentFeatureSet && (
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
                  {currentFeatureSet.features?.map((f, idx) => (
                    <tr key={f.id ?? `feature-${idx}`} className={f.name === "target" ? "font-semibold" : ""}>
                    {/* <tr key={f.id ?? `feature-${idx}`}> */}
                    {/* <tr key={f.id}> */}
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
                        <button
                          className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
                          onClick={() => handleDelete(f.id)}
                        >
                          Delete
                        </button>
                      </td>
{/* 
                      {/* Target Row */}
                      {/* {currentFeatureSet.target && (
                        <tr key={"target"} className="bg-gray-50">
                          <td className="border p-2 font-semibold">Target</td>
                          <td className="border p-2" colSpan="3">
                            {currentFeatureSet.target}
                          </td>
                          <td className="border p-2"></td>
                        </tr>
                      )} */}

                    </tr>
                  ))}

                  {/* Target row */}
                  {/* {currentFeatureSet.target && (
                    <tr key="target" className="border p-2">
                      <td className="border p-2 font-semibold">Target</td>
                      <td className="border p-2">{currentFeatureSet.target.input_data_fields?.join(", ")}</td>
                      <td className="border p-2">{currentFeatureSet.target.function}</td>
                      <td className="border p-2">{JSON.stringify(currentFeatureSet.target.function_parameters)}</td>
                     <td className="border p-2 space-x-2">
                        <button
                          className="bg-blue-500 text-white px-2 py-1 rounded"
                          onClick={() => setEditingFeature(f)}
                          >
                          Edit
                        </button>
                      </td>                    
                    </tr>
                  )} */}

                </tbody>
              </table>
            )}
          </div>
        </div>
      )};
    </div>
  );
}



            {/* // <table className="w-full border-collapse border border-gray-300">
            //   <thead>
            //   {/* <thead className="bg-gray-100"> */}
            {/* //     <tr className="bg-gray-100 text-black">
            //       <th className="border p-2">Name</th>
            //       <th className="border p-2">Input Fields</th>
            //       <th className="border p-2">Function</th>
            //       <th className="border p-2">Parameters</th>
            //       <th className="border p-2">Action</th>
            //     </tr>
            //   </thead>

            //   <tbody>
                // {currentFeatureSet.features?.map((feature, index) => ( */}
            {/* //       <tr key={index}>
            //         <td className="border p-2">{feature.name}</td>
            //         <td className="border p-2">
            //           {feature.input_data_fields?.join(", ")}
            //         </td>
            //         <td className="border p-2">{feature.function}</td>
            //         <td className="border p-2">
            //           {JSON.stringify(feature.function_parameters)}
            //         </td>
            //         <td className="border p-2 space-x-2">
            //           <button */}
            {/* //             className="bg-blue-500 text-white px-2 py-1 rounded"
            //             onClick={() => { */}
            {/* //               console.log("Edit feature:", feature);
            //             }}
            //           >
            //             Edit
            //           </button>
            //           <button */}
            {/* //             className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
            //             onClick={() => { */}
            {/* //               console.log("Delete feature:", feature);
            //             }}
            //           >
            //             Delete
            //           </button>
            //         </td> */}
            {/* //       </tr>
            //     ))} */}

            //     {/* Target Row */}
            {/* //     {currentFeatureSet.target && ( */}
            {/* //       <tr>
            //       {/* <tr className="bg-gray-100 text-black"> */}
            {/* //         <td className="border p-2">{currentFeatureSet.target.name || "Target"}</td>
            //         <td className="border p-2">
            //           {currentFeatureSet.target.input_data_fields?.join(", ")}
            //         </td>
            //         <td className="border p-2">{currentFeatureSet.target.function}</td>
            //         <td className="border p-2">
            //           {JSON.stringify(currentFeatureSet.target.function_parameters)}
            //         </td>
            //         <td className="border p-2 space-x-2">
            //           <button */}
            {/* //             className="bg-blue-500 text-white px-2 py-1 rounded"
            //             onClick={() => { */}
            {/* //               console.log("Edit target:", currentFeatureSet.target);
            //             }}
            //           >
            //             Edit
            //           </button>
            //           <button */}
            {/* //             className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
            //             onClick={() => { */}
            {/* //               console.log("Delete target:", currentFeatureSet.target);
            //             }}
            //           >
            //             Delete
            //           </button>
            //         </td> */}
            {/* //       </tr>
            //     )}

            //     {(!currentFeatureSet.features || */}
            {/* //       currentFeatureSet.features.length === 0) &&
            //       !currentFeatureSet.target && (
            //         <tr>
            //           <td */}
            {/* //             colSpan="5"
            //             className="px-6 py-4 text-center text-gray-500"
            //           >
            //             No features defined
            //           </td>
            //         </tr> */}
            {/* //       )}
            //   </tbody> */}

              {/* <tbody>
              {/* <tbody className="divide-y divide-gray-200"> */}
                {/* {currentFeatureSet.features?.map((feature, index) => (
                  <tr key={index}>
                    <td className="border p-2">{feature.name}</td>
                    <td className="border p-2">{feature.input_data_fields?.join(', ')}</td>
                    <td className="border p-2">{feature.function}</td>
                    <td className="border p-2">{JSON.stringify(feature.function_parameters)}</td>
                  </tr>
                ))}
 */}
                {/* Target Row */}
                {/* {currentFeatureSet.target && (
                  <tr className="bg-gray-50">
                    <td className="border p-2 font-semibold">Target</td>
                    <td className="border p-2" colSpan="3">
                      {currentFeatureSet.target}
                    </td>
                    <td className="border p-2"></td>
                  </tr>
                )}

                {(!currentFeatureSet.features || currentFeatureSet.features.length === 0) && (
                  <tr>
                    <td colSpan="4" className="px-6 py-4 text-center text-gray-500">
                      No features defined
                    </td>
                  </tr>
                )}
              </tbody> */}
            {/* // </table>
          </div>
        </div>
      )}

      // {!currentFeatureSet && (
      //   <div className="text-center text-gray-500 py-8">
      //     Select a feature set to view its features
      //   </div>
      // )} */}

// export default FeatureSetManager;