import { useState, useEffect } from "react";

export const ModelForm = ({
    initialValues = {},
    onSubmit,
    modelNames,
    featureSets,
}) => {
    const [selectedModel, setSelectedModel] = useState(modelNames[0] || "");
    const [selectedFeatureSet, setSelectedFeatureSet] = useState(featureSets[0] || "");
    const [normalise, setNormalise] = useState(false);

    useEffect(() => {
        if (initialValues) {
        setSelectedModel(initialValues.model);
        setSelectedFeatureSet(initialValues.featureSet);
        setNormalise(initialValues.normalise);
        }
    }, [initialValues]);
  
    // useEffect(() => {
    //     setSelectedModel(initialValues.model || modelNames[0] || "");
    //     setSelectedFeatureSet(initialValues.featureSet || featureSets[0] || "");
    //     setNormalise(initialValues.normalise || false);
    // }, [initialValues, modelNames, featureSets]);

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit({
            id: initialValues?.id ?? Date.now(),
            name: `${selectedModel}-${selectedFeatureSet}`,
            model: selectedModel,
            featureSet: selectedFeatureSet,
            normalise,
        });

        // ✅ Reset form fields
        setSelectedModel(modelNames[0] || "");
        setSelectedFeatureSet(featureSets[0] || "");
        setNormalise(false);
    };



//     const handleAddModel = (e) => {
//         e.preventDefault();
//         const newModel = {
//         id: Date.now(), // parent could override with backend id
//         name: `${selectedModel}-${selectedFeatureSet}`,
//         model: selectedModel,
//         featureSet: selectedFeatureSet,
//         normalise,
//         };
//         onAddModel?.(newModel);

//         // reset normalise if you want
//         setNormalise(false);
//   };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
    {/* Select Model */}
    <div>
        <label className="block mb-1">Model</label>
        <select
        className="border rounded p-2 w-auto min-w-[12rem] max-w-xs text-black bg-white"
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
        >
        {modelNames.map((mn) => (
            <option key={mn} value={mn}>
            {mn}
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
        {initialValues ? "Update Model" : "Add Model"}
        {/* Save */}
    </button>
    </form>
  );
};
