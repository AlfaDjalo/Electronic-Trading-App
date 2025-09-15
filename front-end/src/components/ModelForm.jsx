import { useState, useEffect } from "react";

export const ModelForm = ({
    modelConfig,
    featureSets,
    initialValues = {},
    onSubmit
}) => {
    // ensure initialValues is never null
    const safeInitial = initialValues || {};

    const [selectedModel, setSelectedModel] = useState(safeInitial.model || Object.keys(modelConfig)[0] || "");
    const [selectedFeatureSet, setSelectedFeatureSet] = useState(safeInitial.featureSet || featureSets[0] || "");
    const [normalise, setNormalise] = useState(safeInitial.normalise ?? false);
    const [params, setParams] = useState(safeInitial.params || {});
    const [forecastPeriod, setForecastPeriod] = useState(safeInitial.forecastPeriod || 1);
    const [inputWidth, setInputWidth] = useState(safeInitial.inputWidth || 1);

    // Fix: Access parameters directly, not under a 'parameters' property
    const parametersForModel = selectedModel ? modelConfig[selectedModel] || {} : {};

    const handleParamChange = (key, value) => {
        setParams(prev => ({ ...prev, [key]: value }));
    };

    useEffect(() => {
        const iv = initialValues || {};
        setSelectedModel(iv.model || Object.keys(modelConfig)[0] || "");
        setSelectedFeatureSet(iv.featureSet || featureSets[0] || "");
        setNormalise(iv.normalise ?? false);
        setParams(iv.params || {});
        setForecastPeriod(iv.forecastPeriod || 1);
        setInputWidth(iv.inputWidth || 1);
    }, [initialValues, featureSets, modelConfig]);

    const handleSubmit = (e) => {
        e.preventDefault();

        const autoName = `${selectedModel}-${selectedFeatureSet}`;
        
        onSubmit({
            ...safeInitial,
            name: autoName,
            model: selectedModel,
            featureSet: selectedFeatureSet,
            normalise,
            params,
            forecastPeriod,
            inputWidth
        });
        // Reset only in Add mode
        if (!safeInitial.id) {
            setSelectedModel(Object.keys(modelConfig)[0] || "");
            setSelectedFeatureSet(featureSets[0] || "");
            setNormalise(false);
            setParams({});
            setForecastPeriod(1);
            setInputWidth(1);
        }
    };

    return (
        <form
            onSubmit={handleSubmit}
            className="max-w-5xl mx-auto bg-white p-6 rounded-lg shadow-md grid grid-cols-1 md:grid-cols-2 gap-8"
        >
            {/* Left Column: Model selection */}
            <div className="space-y-4">
                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Select Model
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="min-w-[200px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                    >
                        {Object.keys(modelConfig).map((m) => (
                            <option key={m} value={m}>
                                {m}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Feature Set
                    </label>
                    <select
                        value={selectedFeatureSet}
                        onChange={(e) => setSelectedFeatureSet(e.target.value)}
                        className="min-w-[200px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                    >
                        {featureSets.map((f) => (
                            <option key={f} value={f}>
                                {f}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Forecast Period
                    </label>
                    <input
                        type="number"
                        min={1}
                        max={365}   // <-- you can adjust this max as needed
                        step={1}
                        value={forecastPeriod}
                        onChange={(e) => setForecastPeriod(parseInt(e.target.value, 10) || 1)}
                        className="min-w-[150px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Input Width
                    </label>
                    <input
                        type="number"
                        min={1}
                        max={365}   // <-- you can adjust this max as needed
                        step={1}
                        value={inputWidth}
                        onChange={(e) => setInputWidth(parseInt(e.target.value, 10) || 1)}
                        className="min-w-[150px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                    />
                </div>

                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        checked={normalise}
                        onChange={(e) => setNormalise(e.target.checked)}
                        className="h-4 w-4 text-blue-600 border-gray-300 rounded"
                    />
                    <label className="text-sm text-gray-700">Normalise</label>
                </div>
            </div>

            {/* Right Column: Parameters */}
            <div className="space-y-4">
                {selectedModel &&
                    Object.entries(modelConfig[selectedModel] || {}).map(
                        ([param, def]) => (
                            <div key={param}>
                                <label className="block mb-1 text-sm font-medium text-gray-700">
                                    {param}
                                </label>
                                <input
                                    type="text"
                                    value={params[param] ?? def.default}
                                    onChange={(e) =>
                                        setParams({ ...params, [param]: e.target.value })
                                    }
                                    className="min-w-[150px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                                />
                            </div>
                        )
                    )}
            </div>

            {/* Button (spans both columns) */}
            <div className="col-span-1 md:col-span-2 flex justify-center pt-4">
                <button
                    type="submit"
                    className="px-6 py-2 rounded bg-blue-600 text-white font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    {initialValues?.id ? "Update Model" : "Add Model"}
                </button>
            </div>
        </form>
    );
};