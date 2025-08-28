import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState } from "react";
import './App.css'
import "./index.css"

import { Navbar } from "./components/Navbar"
import { MobileMenu } from "./components/MobileMenu"
import { DataUpload } from './components/sections/DataUpload/DataUpload';
import { ViewData } from "./components/ViewData";
import { ModelSelect } from "./components/ModelSelect";
import { EditModel } from "./components/EditModel";
import { DataSummary } from "./components/DataSummary";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [dataInfo, setDataInfo] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [error, setError] = useState(null);
  const [modelList, setModelList] = useState([]);

  const modelNames = ["Baseline", "LSTM", "CNN", "MLP"]
  const featureSets = ["Technical indicators", "Limit Order Book - Full", "Limit Order Book - Lite"]
  // const [selectedFeatures, setSelectedFeatures] = useState([]);

  const addModel = (newModel) => setModelList((prev) => [...prev, newModel])
  const deleteModel = (id) => setModelList((prev) => prev.filter((m) => m.id !== id));
  const updateModel = (updated) =>
    setModelList((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
  // const editModel = (model) => {
  //   console.log("Edit clicked:", model);
  // };
  const setParameters = (model) => {
    console.log("Set parameters for:", model);
  };

  // Handle successful CSV upload
  const handleUploadSuccess = (results, file) => {
    console.log('Upload successful:', results);
    setDataInfo(results.data_info);
    setUploadedFile(file);
    setError(null);
  };

  // Handle upload errors
  const handleUploadError = (errorMessage) => {
    console.error('Upload error:', errorMessage);
    setError(errorMessage);
    setDataInfo(null);
    setUploadedFile(null);
  };

   // Clear all data and start over
  const handleReset = () => {
    setDataInfo(null);
    setUploadedFile(null);
    setSelectedFeatures([]);
  };

  return (
    <Router>
        {/* Always visible */}
        <Navbar menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>
        <MobileMenu menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>

        <Routes>

          {/* Data Upload */}
          <Route
            path="/"
            element={
              <div>
                <DataUpload 
                  onUploadSuccess={(results, file) =>
                    handleUploadSuccess(results, file)
                  }
                  onUploadError={handleUploadError}
                  uploadedFile={uploadedFile}
                />

                {dataInfo && (
                  <div className="mt-8">
                    Data Summary
                    <DataSummary dataInfo={dataInfo} />
                  </div>
                )}
              </div>
            }
          />

          {/* View Data */}
          <Route
            path="/view_data"
            element={
              dataInfo ? (
                <ViewData
                  chartData={dataInfo.data}
                  featureNames={dataInfo.feature_names}
                />
              ) : (
                <p className="text-center mt-10">Please upload data first.</p>                
              )
            }
          />

          {/* Select Model */}
          {/* <Route path="/load_data" element={<DataUpload />} /> */}
          {/* <Route path="/select_model" element={<SelectModel />} /> */}
          {/* <Route path="/display_results" element={<DisplayResults />} /> */}
          <Route
            path="/select_model"
            element={
              dataInfo ? (
                <ModelSelect
                  modelNames={modelNames}
                  featureSets={featureSets}
                  modelList={modelList}
                  onAddModel={addModel}
                  onDelete={deleteModel}
                  onEdit={updateModel}
                  onSetParameters={setParameters}
                />
              ):(
                <p className="text-center mt-10">
                  Please upload data first.
                </p>
              )
            }
          />

          {/* <Route path="/load_data" element={<DataUpload />} /> */}
          {/* <Route
            path="/models/:id/edit"
            element={
              dataInfo ? (
                <EditModel
                  modelNames={modelNames}
                  featureSets={featureSets}
                  onSave={updateModel}
                  modelList={modelList}
                />
              ):(
                <p className="text-center mt-10">Please upload data first.</p>
              )
            }
          /> */}

        </Routes>
    </Router>
  );
}

export default App
