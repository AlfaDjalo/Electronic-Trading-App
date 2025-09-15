import React, { useState, useRef, useEffect } from 'react';
import './DataUpload.css';
// import { TickerFetcher } from "./TickerFetcher";

export const DataUpload = ({ onUploadSuccess, onUploadError, uploadedFileName }) => {
  const [mode, setMode] = useState("csv");
  const [category, setCategory] = useState("");
  const [tickers, setTickers] = useState([]);
  const [ticker, setTicker] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");

  const [selectedFileName, setSelectedFileName] = useState(uploadedFileName || null);
  const [uploadStatus, setUploadStatus] = useState(uploadedFileName ? "success" : "idle");
  const [errorMessage, setErrorMessage] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (uploadedFileName) {
      setSelectedFileName(uploadedFileName);
      setUploadStatus("success");
    } else {
      setSelectedFileName(null);
      setUploadStatus("idle");
    }
  }, [uploadedFileName]);
  
  useEffect(() => {
    console.log(category)
    if (category) {
      fetch(`http://localhost:5000/api/get_tickers/${category}`)
        .then((res) => res.json())
        .then((data) => setTickers(data.tickers || []))
        .catch((err) => console.error("Error fetching tickers:", err));
      }
  }, [category]);

  useEffect(() => {
    console.log("📊 tickers state updated:", tickers);
  }, [tickers]);

  // FILE VALIDATION FUNCTION
  // This runs on the client side before uploading to catch obvious issues early
  const validateFile = (fileName) => {
    // Check if a file was actually selected
    if (!fileName) {
      return "Please select a file";
    }

    // Validate file type - only allow CSV files
    // file.type might be empty on some systems, so we also check the extension
    if (fileName.type !== 'text/csv' && !fileName.name.toLowerCase().endsWith('.csv')) {
      return "Please select a CSV file";
    }

    // Check file size - prevent huge files that might crash the browser/server
    // 10MB limit (10 * 1024 * 1024 bytes)
    const maxSize = 10 * 1024 * 1024; 
    if (fileName.size > maxSize) {
      return "File size must be less than 10MB";
    }

    // File passed all validation checks
    return null;
  };


  // FILE SELECTION HANDLER
  // Called when user selects a file through the file input or drag & drop
  const handleFileSelect = (fileName) => {
    // Validate the selected file
    const validationError = validateFile(fileName);
    if (validationError) {
      setErrorMessage(validationError);
      setSelectedFileName(null);
      setUploadStatus('error');
      return;
    }

    // File is valid, store it and clear any previous errors
    setSelectedFileName(fileName);
    setErrorMessage('');
    setUploadStatus('idle');
    setUploadProgress(0);

    // Automatically trigger upload
    handleUpload(fileName);
  };

  // FILE INPUT CHANGE HANDLER
  // Triggered when user uses the file picker dialog
  const handleFileInputChange = (event) => {
    const fileName = event.target.files[0];
    handleFileSelect(fileName);
  };

  // DRAG AND DROP HANDLERS
  // These provide a more modern UX for file selection
  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDragEnter = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // UPLOAD FUNCTION
  // This sends the file to your Python backend API
  const handleUpload = async (fileToUpload) => {
    const file = fileToUpload || selectedFileName;
    if (!file) {
      setErrorMessage("Please select a file first");
      return;
    }

    // Update status to show upload is starting
    setUploadStatus('uploading');
    setErrorMessage('');
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('fileName', file);

      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setUploadProgress(Math.round(percentComplete));
        }
      });

      xhr.onload = function() {
        if (xhr.status === 200) {
          const response = JSON.parse(xhr.responseText);
          setUploadStatus('success');
          setUploadProgress(100);
          if (onUploadSuccess) {
            onUploadSuccess(response, file);
          }
        } else {
          setUploadStatus('error');
          setErrorMessage(`Upload failed: ${xhr.status} ${xhr.statusText}`);
          if (onUploadError) {
            onUploadError(`Upload failed: ${xhr.status} ${xhr.statusText}`);
          }
        }
      };

      xhr.onerror = function() {
        setUploadStatus('error');
        setErrorMessage('Network error occurred during upload');
        if (onUploadError) {
          onUploadError('Network error occurred during upload');
        }
      };

      xhr.open('POST', 'http://localhost:5000/api/upload_data', true);
      xhr.send(formData);

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus('error');
      setErrorMessage('An unexpected error occurred');
      if (onUploadError) {
        onUploadError('An unexpected error occurred');
      }
    }
  };

  // CLEAR SELECTION FUNCTION
  // Allows user to start over with a different file
  const handleClear = () => {
    setSelectedFileName(null);
    setUploadStatus('idle');
    setErrorMessage('');
    setUploadProgress(0);
    
    // Clear the file input value
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleYahooSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch("http://localhost:5000/api/yahoo_data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker, startDate, endDate }),
      });
      const json = await response.json();
      if (json.success) {
        onUploadSuccess?.(json, "yahoo");
      } else {
        setErrorMessage(json.error || "Failed to load Yahoo Finance data");
      }
    } catch (err) {
      console.error("Yahoo load error:", err);
      setErrorMessage("Network error while loading Yahoo Finance data");
    }
  };

  // FORMAT FILE SIZE FOR DISPLAY
  // Converts bytes to human-readable format
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // COMPONENT RENDER
  return (
    <div className="data-upload">
      <h2>Load Data</h2>

      {/* Toggle buttons */}
      <div className="toggle-buttons">
        <button
          className={mode === "csv" ? "active" : ""}
          onClick={() => setMode("csv")}
        >
          Load CSV File
        </button>
        <button
          className={mode === "yahoo" ? "active" : ""}
          onClick={() => setMode("yahoo")}
        >
          Load from Yahoo Finance
        </button>       
      </div>

      {mode === "csv" ? (
        <>
          <p>Select a CSV file containing time series returns for your features</p>

          {/* DRAG & DROP AREA */}
          <div 
            className={`upload-area ${uploadStatus === 'uploading' ? 'uploading' : ''}`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()} // Click to open file picker
          >
            {/* HIDDEN FILE INPUT */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".csv"
              style={{ display: 'none' }}
            />

            {/* UPLOAD AREA CONTENT */}
            {!selectedFileName ? (
              <div className="upload-prompt">
                <div className="upload-icon">📁</div>
                <p>Click to select a CSV file or drag and drop here</p>
                <p className="upload-hint">Maximum file size: 10MB</p>
              </div>
            ) : (
              <div className="file-selected">
                <div className="file-icon">📄</div>
                <div className="file-info">
                  <p className="file-name">{selectedFileName.name}</p>
                  <p className="file-size">{formatFileSize(selectedFileName.size)}</p>
                  <p className={`file-status ${uploadStatus}`}>
                    {uploadStatus === 'success' && '✅ Uploaded successfully'}
                    {uploadStatus === 'uploading' && `⏳ Uploading... ${uploadProgress}%`}
                    {uploadStatus === 'error' && `⚠️ Error: ${errorMessage}`}
                    {uploadStatus === 'idle' && '📄 Ready to upload'}
                  </p>
                </div>
                  {uploadStatus !== 'uploading' && (
                    <button 
                    className="clear-button"
                    onClick={(e) => {
                      e.stopPropagation(); // Prevent triggering the file picker
                      handleClear();
                      }}
                      >
                      ✕
                    </button>
                  )}
              </div>
            )}
          </div>
        </>
      ) : (
        <>
          {/* Yahoo Finance Data */}
          <form onSubmit={handleYahooSubmit} className="p-4 border rounded space-y-4">
            <h3 className="font-semibold mb-2">Fetch Yahoo Data</h3>

            {/* <TickerFetcher
              category={category}
              onCategoryChange={setCategory}
              ticker={ticker}
              onTickerChange={setTicker}
              onTickersLoaded={setTickers}
            /> */}

            <div>
              <label className="block">Category:</label>
              <select
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="border p-2 rounded w-full bg-white text-black"
              >
                <option value="">Select Category</option>
                <option value="australian">Australian (ASX200)</option>
                <option value="us">US (S&P 500)</option>
                <option value="fx">FX</option>
                <option value="crypto">Crypto</option>
                <option value="test">Test</option>
              </select>
            </div>

            <div>
              <label className="block">Ticker:</label>
              <select
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                className="border p-2 rounded w-full bg-white text-black"
                disabled={!tickers.length}
              >
                <option value="">Select Ticker</option>
                {tickers
                  .filter((t) => t.Code && t.Company)
                  .map((t, idx) => (
                    <option key={`${t.Code}-${idx}`} value={t.Code}>
                      {t.Code} – {t.Company}
                    </option>
                  ))}
              </select>
            </div>

            <div>
              <label className="block">Start Date:</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="border p-2 rounded w-full bg-white text-black"
              />
            </div>

            <div>
              <label className="block">End Date:</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="border p-2 rounded w-full bg-white text-black"
              />
            </div>

            <button
              type="submit"
              className="px-6 py-2 bg-green-600 text-white rounded hover:bg-green-700"
            >
              Fetch Data
            </button>
          </form>  
        </>
      )}

      {/* UPLOAD PROGRESS BAR */}
      {/* {uploadStatus === 'uploading' && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
              ></div>
          </div>
          <p>Uploading... {uploadProgress}%</p>
        </div>
      )} */}

      {/* ERROR MESSAGE DISPLAY */}
      {uploadStatus === 'error' && errorMessage && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {errorMessage}
        </div>
      )}

      {/* SUCCESS MESSAGE */}
      {/* {uploadStatus === 'success' && (
        <div className="success-message">
          <span className="success-icon">✅</span>
          File uploaded and processed successfully!
        </div>
      )} */}

      {/* ACTION BUTTONS */}
      {/* <div className="button-container"> */}
        {/* Remove Upload button */}
        {/* Only show "Choose Different File" if a file is selected and not uploading */}
        {/* {selectedFileName && uploadStatus !== 'uploading' && (
          <button 
          className="clear-button-secondary"
          onClick={handleClear}
          >
            Choose Different File
          </button>
        )}
      </div> */}
    </div>
  );
};