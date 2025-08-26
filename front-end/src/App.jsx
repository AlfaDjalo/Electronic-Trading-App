import { useState } from "react";
import { Navbar } from "./components/Navbar"
import { MobileMenu } from "./components/MobileMenu"
import './App.css'
import "./index.css"
import DataUpload from './components/sections/DataUpload/DataUpload';


function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);

  // Handle successful CSV upload
  const handleUploadSuccess = (results) => {
    console.log('Upload successful:', results);
    setDataInfo(results.data_info);
    setOptimizationResults(results);
    setError(null);
  };

  // Handle upload errors
  const handleUploadError = (errorMessage) => {
    console.error('Upload error:', errorMessage);
    setError(errorMessage);
    setDataInfo(null);
    setOptimizationResults(null);
  };

   // Clear all data and start over
  const handleReset = () => {
    // setOptimizationResults(null);
    setDataInfo(null);
    // setOptimizationData(null);
    // setSelectedMethods([]);
    // setIsProcessing(false);
    // setError(null);
  };

  return (
    <>
      <div className={"min-h-screen bg-black text-gray-100"}>
        <Navbar menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>
        <MobileMenu menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>
        Hello World !

        {/* Step 1: Data Upload */}
        <div className="step-container">
          <div className="step-header">
            <span className="step-number">1</span>
            <h2>Upload Data</h2>
          </div>
          
          <DataUpload 
            onUploadSuccess={handleUploadSuccess}
            onUploadError={handleUploadError}
          />

          {/* Show data summary if upload successful */}
          {dataInfo && (
            <div className="data-summary">
              <h3>Data Successfully Loaded</h3>
              <div className="summary-grid">
                <div className="summary-item">
                  <span className="label">Assets:</span>
                  <span className="value">{dataInfo.num_assets}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Observations:</span>
                  <span className="value">{dataInfo.num_observations}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Date Range:</span>
                  <span className="value">{dataInfo.date_range}</span>
                </div>
                <div className="summary-item">
                  <span className="label">Assets:</span>
                  <span className="value">{dataInfo.asset_names.join(', ')}</span>
                </div>
              </div>
              <button className="reset-btn" onClick={handleReset}>
                Upload Different Data
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App
