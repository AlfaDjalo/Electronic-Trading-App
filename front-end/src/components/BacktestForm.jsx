import { useState, useEffect } from "react";

export const BacktestForm = ({
    modelList, 
    initialValues={},
    onSubmit,
    onCancel
}) => {
    const safeInitial = initialValues || {};
    const [selectedModel, setSelectedModel] = useState(
        safeInitial.modelName || modelList[0]?.name || ""
    )
    const selectedModelObj = modelList.find(m => m.name === selectedModel);

    // Backtesting parameters (shared across all models)
    const [initialCash, setInitialCash] = useState(safeInitial.initialCash || 100000);
    const [transactionCost, setTransactionCost] = useState(safeInitial.transactionCost || 0.0005);
    const [tradeThreshold, setTradeThreshold] = useState(safeInitial.tradeThreshold || 0.0002);
    const [positionSizing, setPositionSizing] = useState(safeInitial.positionSizing || "full");
    const [maxPositions, setMaxPositions] = useState(safeInitial.maxPositions || 1);
    const [delay, setDelay] = useState(safeInitial.delay || 1);
    const [stopLoss, setStopLoss] = useState(safeInitial.stopLoss || 0);
    const [takeProfit, setTakeProfit] = useState(safeInitial.takeProfit || 0);


    useEffect(() => {
        // Reset form when initialValues change
        const iv = initialValues || {};

        if (iv.modelName) {
            setSelectedModel(iv.modelName);
        }

        setInitialCash(iv.initialCash || 1000);
        setTransactionCost(iv.transactionCost || 0.0005);
        setTradeThreshold(iv.tradeThreshold || 0.01);
        setPositionSizing(iv.positionSizing || "full");
        setMaxPositions(iv.maxPositions || 1);
        setDelay(iv.delay || 1);
        setStopLoss(iv.stopLoss || -20);
        setTakeProfit(iv.takeProfit || 20);
    }, [initialValues]);

    const handleSubmit = (e) => {
        e.preventDefault();
        
        const data = {
            ...safeInitial,
            modelID: selectedModelObj.id,
            modelName: selectedModel,
            backtestParams: {
                initialCash,
                transactionCost,
                tradeThreshold,
                positionSizing,
                maxPositions,
                delay,
                stopLoss,
                takeProfit                
            }
        }
    
        console.log("Submitting backtest:", data);
        onSubmit(data);
    };

    return (
        <form
            onSubmit={handleSubmit}
            className="max-w-5xl mx-auto bg-white p-6 rounded-lg shadow-md grid grid-cols-1 md:grid-cols-2 gap-8"
        >
            <div className="space-y-4">
                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Model                        
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="min-w-[200px] border border-gray-300 rounded px-3 py-2 text-black bg-white focus:ring-2 focus:ring-blue-500"
                    >
                        {modelList.map((m) => (
                            <option key={m.id} value={m.name}>
                                {m.name}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Feature Set
                    </label>
                    <input
                        type="text"
                        value={selectedModelObj?.featureSet || ""}
                        disabled
                        className="w-full border border-gray-300 rounded px-3 py-2 bg-gray-100 text-gray-600"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Forecast Period
                    </label>
                    <input
                        type="number"
                        value={selectedModelObj?.forecastPeriod || ""}
                        disabled
                        className="w-full border border-gray-300 rounded px-3 py-2 bg-gray-100 text-gray-600"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Input Width
                    </label>
                    <input
                        type="number"
                        value={selectedModelObj?.inputWidth || ""}
                        disabled
                        className="w-full border border-gray-300 rounded px-3 py-2 bg-gray-100 text-gray-600"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Normalise
                    </label>
                    <input
                        type="text"
                        value={selectedModelObj?.normalise ? "Yes" : "No"}
                        disabled
                        className="w-full border border-gray-300 rounded px-3 py-2 bg-gray-100 text-gray-600"
                    />
                </div>
            </div>

            <div className="space-y-4">
                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Initial Cash
                    </label>
                    <input
                        type="number"
                        value={initialCash}
                        onChange={(e) => setInitialCash(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Transaction Cost (fraction)
                    </label>
                    <input
                        type="number"
                        step="0.0001"
                        value={transactionCost}
                        onChange={(e) => setTransactionCost(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Trade Threshold (fraction)
                    </label>
                    <input
                        type="number"
                        step="0.0001"
                        value={tradeThreshold}
                        onChange={(e) => setTradeThreshold(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Position Sizing
                    </label>
                    <select
                        value={positionSizing}
                        onChange={(e) => setPositionSizing(e.target.value)}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    >
                        <option value="full">Full Allocation</option>
                        <option value="half">Half Allocation</option>
                        <option value="fixed_percent">Fixed % of Portfolio</option>
                    </select>
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Max Positions
                    </label>
                    <input
                        type="number"
                        min={1}
                        value={maxPositions}
                        onChange={(e) => setMaxPositions(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Execution Delay (bars)
                    </label>
                    <input
                        type="number"
                        min={0}
                        value={delay}
                        onChange={(e) => setDelay(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Stop Loss (%)
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        value={stopLoss}
                        onChange={(e) => setStopLoss(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>

                <div>
                    <label className="block mb-1 text-sm font-medium text-gray-700">
                        Take Profit (%)
                    </label>
                    <input
                        type="number"
                        step="0.1"
                        value={takeProfit}
                        onChange={(e) => setTakeProfit(Number(e.target.value))}
                        className="w-full border border-gray-300 rounded px-3 py-2 text-black bg-white"
                    />
                </div>
            </div>
            <div className="col-span-1 md:col-span-2 flex justify-center pt-4">
                <button
                    type="submit"
                    className="px-6 py-2 rounded bg-blue-600 text-white font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    {initialValues?.id ? "Update Backtest" : "Add Backtest"}
                </button>
                <button
                    type="button"
                    onClick={() => {
                        setSelectedModel(modelList[0]?.name || "");
                        // setBacktestParams({});
                        setInitialCash(100000);
                        setTransactionCost(0.0005);
                        setTradeThreshold(0.01);
                        setPositionSizing("full");
                        setMaxPositions(1);
                        setDelay(1);
                        setStopLoss(0);
                        setTakeProfit(0);
                        onCancel?.();
                    }}
                    className="px-6 py-2 rounded bg-blue-600 text-white font-medium hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                    Cancel
                </button>
            </div>
        </form>
    )
}