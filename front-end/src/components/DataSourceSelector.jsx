import React, { useState, useEffect } from "react";

export const DataSourceSelector = ({ onUploadSuccess }) => {
  const [mode, setMode] = useState("file"); // "file" or "yahoo"
  const [category, setCategory] = useState("australian");
  const [tickers, setTickers] = useState([]);
  const [ticker, setTicker] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  // Fetch tickers when category changes
  useEffect(() => {
    if (mode === "yahoo") {
      fetch(`/get_tickers/${category}`)
        .then((res) => res.json())
        .then((data) => {
          setTickers(data.tickers);
          if (data.tickers.length > 0) {
            setTicker(data.tickers[0].Code);
          }
        })
        .catch((err) => console.error("Error fetching tickers:", err));
    }
  }, [category, mode]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("fileName", file);

    const response = await fetch("/api/upload_data", {
      method: "POST",
      body: formData,
    });

    const json = await response.json();
    if (json.success) {
      onUploadSuccess(json);
    } else {
      alert("Upload failed: " + json.error);
    }
  };

  const handleYahooDownload = async () => {
    const response = await fetch("/api/yahoo_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, ticker, start_date: startDate, end_date: endDate }),
    });

    const json = await response.json();
    if (json.success) {
      onUploadSuccess(json);
    } else {
      alert("Yahoo data fetch failed: " + json.error);
    }
  };

  return (
    <div className="p-4 border rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Load Data</h2>

      {/* Mode Selector */}
      <div className="mb-4">
        <label className="mr-4">
          <input
            type="radio"
            value="file"
            checked={mode === "file"}
            onChange={() => setMode("file")}
          />
          Upload File
        </label>
        <label className="ml-4">
          <input
            type="radio"
            value="yahoo"
            checked={mode === "yahoo"}
            onChange={() => setMode("yahoo")}
          />
          Yahoo Finance
        </label>
      </div>

      {/* File Upload */}
      {mode === "file" && (
        <div>
          <input type="file" accept=".csv" onChange={handleFileUpload} />
        </div>
      )}

      {/* Yahoo Finance Options */}
      {mode === "yahoo" && (
        <div className="space-y-4">
          {/* Category */}
          <div>
            <label className="block font-medium">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="border px-2 py-1 rounded text-black bg-white"
            >
              <option value="australian">Australian Stocks</option>
              <option value="us">US Stocks</option>
              <option value="fx">Forex</option>
              <option value="crypto">Crypto</option>
            </select>
          </div>

          {/* Ticker */}
          <div>
            <label className="block font-medium">Ticker</label>
            <select
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="border px-2 py-1 rounded text-black bg-white"
            >
              {tickers.map((t) => (
                <option key={t.Code} value={t.Code}>
                  {t.Company}
                </option>
              ))}
            </select>
          </div>

          {/* Dates */}
          <div>
            <label className="block font-medium">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="border px-2 py-1 rounded text-black bg-white"
            />
          </div>
          <div>
            <label className="block font-medium">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="border px-2 py-1 rounded text-black bg-white"
            />
          </div>

          {/* Submit */}
          <button
            onClick={handleYahooDownload}
            className="bg-blue-600 text-white px-4 py-2 rounded"
          >
            Download from Yahoo Finance
          </button>
        </div>
      )}
    </div>
  );
};
