import { useState } from "react";
import { Navbar } from "./components/Navbar"
import { MobileMenu } from "./components/MobileMenu"
import './App.css'
import "./index.css"
import DataUpload from './components/sections/DataUpload/DataUpload';
import { ChartArea } from "./components/ChartArea";
import { SelectModel } from "./components/sections/SelectModel";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [error, setError] = useState(null);
  const [models, setModels] = useState([]);

  const modelNames = ["Baseline", "LSTM", "CNN", "MLP"]
  const featureSets = ["Technical indicators", "Limit Order Book - Full", "Limit Order Book - Lite"]
  // const [selectedFeatures, setSelectedFeatures] = useState([]);

  const addModel = (newModel) => setModels((prev) => [...prev, newModel])
  const deleteModel = (id) => setModels((prev) => prev.filter((m) => m.id !== id));
  const editModel = (model) => {
    console.log("Edit clicked:", model);
  };
  const setParameters = (model) => {
    console.log("Set parameters for:", model);
  };

  // Handle successful CSV upload
  const handleUploadSuccess = (results) => {
    console.log('Upload successful:', results);
    setDataInfo(results.data_info);
    setError(null);
  };

  // Handle upload errors
  const handleUploadError = (errorMessage) => {
    console.error('Upload error:', errorMessage);
    setError(errorMessage);
    setDataInfo(null);
  };

   // Clear all data and start over
  const handleReset = () => {
    setDataInfo(null);
    setSelectedFeatures([]);
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
                  <span className="label">Features:</span>
                  <span className="value">{dataInfo.num_features}</span>
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
                  <span className="label">Features:</span>
                  <span className="value">{dataInfo.feature_names.join(', ')}</span>
                </div>
              </div>
              <button className="reset-btn" onClick={handleReset}>
                Upload Different Data
              </button>
            </div>
          )}

          {/* Feature selection + chart rendering */}
          {dataInfo && (
            <>
              <ChartArea
                chartData={dataInfo.data}
                featureNames={dataInfo.feature_names}
                // selectedFeatures={selectedFeatures}
              />
            </>
          )}

          {/* 
          { dataInfo && <CsvChart data={dataInfo.data} columns={dataInfo.asset_names}/> }
          <ViewData /> */}

          {dataInfo && (
            <div className="max-w-3xl mx-auto">
              <SelectModel
                modelNames={modelNames}
                featureSets={featureSets}
                modelList={models}
                onAddModel={addModel}
                onDelete={deleteModel}
                onEdit={editModel}
                onSetParameters={setParameters}
              />
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App
