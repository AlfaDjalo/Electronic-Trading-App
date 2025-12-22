import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { BacktestForm } from "./BacktestForm";
import { BacktestRun } from "./BacktestRun";

export const BacktestSelect = ({
    modelList,
    backtestList,
    results,
    onAddBacktest,
    onUpdateBacktest,
    onDeleteBacktest,
    setBacktestResults,
}) => {
    const [editingBacktest, setEditingBacktest] = useState(null);
    const navigate = useNavigate();

    const handleAdd = (newBT) => {
        const withId = { ...newBT, id: newBT.id ?? Date.now() };
        onAddBacktest(withId);
        setEditingBacktest(null);
    };

    const handleUpdate = (updatedBT) => {
        onUpdateBacktest(updatedBT);
        setEditingBacktest(null);
    };

    return (
        <div className="p-4">
            <BacktestForm
                modelList={modelList}
                initialValues={editingBacktest}
                onSubmit={editingBacktest ? handleUpdate : handleAdd}
            />

            <hr className="my=6"/>

            <h4 className="text-lg font-semibold mb-3">Existing Backtests</h4>

            {console.log("BacktestSelect: backtestList =", JSON.stringify(backtestList, null, 2))}
            {console.log("BacktestSelect: modelList =", JSON.stringify(modelList, null, 2))}

            {backtestList.length === 0 ? (
                <p className="text-gray-500">No backtests yet.</p>
            ) : (
                <table className="w-full border-collapse border border-gray-300 text-black">
                            <thead>
                                <tr className="bg-gray-100">
                                    <th className="border p-2">Model Name</th>
                                    <th className="border p-2">Backtest Parameters</th>
                                    <th className="border p-2">Actions</th>
                                </tr>
                            </thead>

                    <tbody>
                        {backtestList.map((bt) => (
                            <tr key={bt.id} className="text-white">
                                <td className="border p-2">
                                    {modelList.find(m => m.id === bt.modelID)?.name || "(unknown model)"}
                                </td>

                                <td className="border p-2">
                                    {Object.entries(bt.backtestParams || {})
                                    .map(([k, v]) => `${k}: ${v}`)
                                    .join(", ") || "(none)"}
                                </td>

                                <td className="border p-2 space-x-2">
                                <button
                                    className="bg-blue-500 text-white px-2 py-1 rounded"
                                    onClick={() => setEditingBacktest(bt)}
                                >
                                    Edit
                                </button>

                                <button
                                    className="bg-white text-red-600 px-2 py-1 rounded border border-red-600"
                                    onClick={() => onDeleteBacktest?.(bt.id)}
                                >
                                    Delete
                                </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}

            {backtestList.length > 0 && (
                <div className="flex justify-center mt-6">
                <button
                    className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500"
                    onClick={async () => {
                    try {
                        // Call backend to run backtests
                        const backtestResults = await BacktestRun(modelList, backtestList, results);
                        // Save to app state via setter provided by App
                        if (setBacktestResults) setBacktestResults(backtestResults);
                        // Navigate to results page
                        navigate("/view_backtests");
                    } catch (err) {
                        console.error("Error running backtests:", err);
                    }
                    }}
                >
                    Run Backtests
                </button>
                </div>
            )}
        </div>
    )
}

