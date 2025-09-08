import React, { useState, useRef, useEffect } from 'react';
import './DataUpload.css';

export const DataUpload = ({ onUploadSuccess, onUploadError, uploadedFileName }) => {
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
  };

  // FILE INPUT CHANGE HANDLER
  // Triggered when user uses the file picker dialog
  const handleFileInputChange = (event) => {
    const fileName = event.target.files[0]; // Get the first (and only) selected file
    handleFileSelect(fileName);
  };

  // DRAG AND DROP HANDLERS
  // These provide a more modern UX for file selection
  
  const handleDragOver = (event) => {
    event.preventDefault(); // Prevent default behavior (opening file in browser)
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
    
    // Get the dropped files
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]); // Only take the first file
    }
  };

  // UPLOAD FUNCTION
  // This sends the file to your Python backend API
  const handleUpload = async () => {
    if (!selectedFileName) {
      setErrorMessage("Please select a file first");
      return;
    }

    // Test from ChatGPT
    async function uploadCSV(fileName) {
      const formData = new FormData();
      formData.append("fileName", fileName); // key MUST match Flask: "file"

      // const res = await fetch("http://localhost:5000/api/upload-csv", {
      const res = await fetch("http://localhost:5000/api/upload_data", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      console.log("Upload result:", data);
    }

    // Update status to show upload is starting
    setUploadStatus('uploading');
    setErrorMessage('');
    setUploadProgress(0);

    try {
      // Create FormData object to send file as multipart/form-data
      // This is the standard way to upload files via HTTP
      const formData = new FormData();
      formData.append('fileName', selectedFileName); // 'fileName' is the key your Python API will look for

      // Create XMLHttpRequest to track upload progress
      // fetch() doesn't support upload progress tracking
      const xhr = new XMLHttpRequest();

      // Set up progress tracking
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
          const percentComplete = (event.loaded / event.total) * 100;
          setUploadProgress(Math.round(percentComplete));
        }
      });

      // Set up the response handlers
      xhr.onload = function() {
        if (xhr.status === 200) {
          // Success! Parse the JSON response
          const response = JSON.parse(xhr.responseText);
          setUploadStatus('success');
          setUploadProgress(100);
          
          // Call the parent component's success handler with the results
          // This is how we pass the optimization results up to the main app
          if (onUploadSuccess) {
            onUploadSuccess(response, selectedFileName);
          }
        } else {
          // HTTP error status
          setUploadStatus('error');
          setErrorMessage(`Upload failed: ${xhr.status} ${xhr.statusText}`);
          
          if (onUploadError) {
            onUploadError(`Upload failed: ${xhr.status} ${xhr.statusText}`);
          }
        }
      };

      xhr.onerror = function() {
        // Network error
        setUploadStatus('error');
        setErrorMessage('Network error occurred during upload');
        
        if (onUploadError) {
          onUploadError('Network error occurred during upload');
        }
      };

      // Send the request to your Python API
      // TODO: Replace with your actual API endpoint URL
      // xhr.open('POST', 'http://localhost:5000/api/upload-csv', true);
      xhr.open('POST', 'http://localhost:5000/api/upload_data', true);
      xhr.send(formData);

    } catch (error) {
      // Unexpected error
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
      <h2>Upload Portfolio Data</h2>
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

      {/* UPLOAD PROGRESS BAR */}
      {uploadStatus === 'uploading' && (
        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
          <p>Uploading... {uploadProgress}%</p>
        </div>
      )}

      {/* ERROR MESSAGE DISPLAY */}
      {uploadStatus === 'error' && errorMessage && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          {errorMessage}
        </div>
      )}

      {/* SUCCESS MESSAGE */}
      {uploadStatus === 'success' && (
        <div className="success-message">
          <span className="success-icon">✅</span>
          File uploaded and processed successfully!
        </div>
      )}

      {/* ACTION BUTTONS */}
      <div className="button-container">
        <button 
          className="upload-button"
          onClick={handleUpload}
          disabled={!selectedFileName || uploadStatus === 'uploading'}
        >
          {uploadStatus === 'uploading' ? 'Processing...' : 'Upload'}
        </button>

        {selectedFileName && uploadStatus !== 'uploading' && (
          <button 
            className="clear-button-secondary"
            onClick={handleClear}
          >
            Choose Different File
          </button>
        )}
      </div>
    </div>
  );
};