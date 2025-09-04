import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState, useEffect } from "react";
import './App.css'
import "./index.css"

import { Navbar } from "./components/Navbar"
import { MobileMenu } from "./components/MobileMenu"
import { DataUpload } from './components/sections/DataUpload/DataUpload';
import { ViewData } from "./components/ViewData";
import { ViewResults } from "./components/ViewResults";
import { ModelSelect } from "./components/ModelSelect";
// import { DataSummary } from "./components/DataSummary";
import { useModelConfig } from "./hooks/useModelConfig";
import { ProcessData } from "./components/ProcessData";
import { ModelRun } from "./components/ModelRun";
// import { DisplayResults } from "./components/DisplayResults";

function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [rawData, setRawData] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [modelList, setModelList] = useState([]);
  const [results, setResults] = useState(null);
  const [seriesNames, setSeriesNames] = useState([]);
  const [featureSet, setFeatureSet] = useState(null);
  const [error, setError] = useState(null);
  
  const modelConfig = useModelConfig();
  
  if (!modelConfig) return <p>Loading model config...</p>;

  const modelNames = Object.keys(modelConfig);
  // const modelNames = ["Baseline", "LSTM", "CNN", "MLP"]
  const featureSets = ["Limit_Order_Book", "Cut_Down_Limit_Order_Book", "Sandbox_Intraday"]
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
  const handleUploadSuccess = (results, fileName) => {
    console.log('Upload successful:', results.success);

    const timeSeriesData = results.timeSeriesData;
    setRawData(timeSeriesData);
    setUploadedFileName(fileName);

    if (timeSeriesData && timeSeriesData.length > 0) {
      const names = Object.keys(timeSeriesData[0]).filter(k => k !== "date");
      setSeriesNames(names);
    } else {
      console.warn("No rows found in upload results");
    }

    setError(null);
  };

  // Handle upload errors
  const handleUploadError = (errorMessage) => {
    console.error('Upload error:', errorMessage);
    setError(errorMessage);
    setRawData(null);
    setUploadedFileName(null);
  };

   // Clear all data and start over
  const handleReset = () => {
    setRawData(null);
    setUploadedFileName(null);
    setSelectedFeatures([]);
  };

  // Handle successful CSV upload
  const handleProcessSuccess = (results) => {
    console.log('Processing successful:', results.success);

    const timeSeriesData = results.timeSeriesData;
    setProcessedData(timeSeriesData);

    // if (timeSeriesData && timeSeriesData.length > 0) {
    //   const names = Object.keys(timeSeriesData[0]).filter(k => k !== "date");
    //   setSeriesNames(names);
    // } else {
    //   console.warn("No rows found in upload results");
    // }

    setError(null);
  };

  // Handle upload errors
  const handleProcessError = (errorMessage) => {
    console.error('Processing error:', errorMessage);
    setError(errorMessage);
    setProcessedData(null);
  };

  const runModels = async (modelList, rawData) => {
    try {
      const res = await ModelRun(modelList, rawData);
      setResults(res);
    } catch (err) {
      console.error("Model run failed:", err);
    }
  };

  return (
    <Router>
        {/* Always visible */}
        <Navbar menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>
        <MobileMenu menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>

        <div className="pt-16">
          <Routes>

            {/* Data Upload */}
            <Route
              path="/"
              element={
                <div>
                  <DataUpload 
                    onUploadSuccess={(results, fileName) =>
                      handleUploadSuccess(results, fileName)
                    }
                    onUploadError={handleUploadError}
                    uploadedFileName={uploadedFileName}
                  />
                </div>
              }
            />

            {/* View Data */}
            <Route
              path="/view_data"
              element={
                rawData ? (
                  <ViewData
                    data={rawData}
                    seriesNames={seriesNames}
                    // chartData={dataInfo.data}
                    // featureNames={dataInfo.feature_names}
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
                rawData ? (
                  <div className="pt-20">   {/* <-- offset for navbar height */}
                    <ModelSelect
                      modelNames={modelNames}
                      modelConfig={modelConfig}
                      featureSets={featureSets}
                      modelList={modelList}
                      onAddModel={addModel}
                      onDelete={deleteModel}
                      onEdit={updateModel}
                      rawData={rawData}
                      setResults={setResults}
                    />
                  </div>
                ):(
                  <p className="text-center mt-10">
                    Please upload data first.
                  </p>
                )
              }
            />

          {/* <Route
            path="/view_processed_data"
            element={
              processedData ? (
                <ViewData
                  data={processedData}
                  seriesNames={featureSets}
                />
              ) : featureSet.length > 0 && rawData ? (
                <ProcessData
                  onProcessSuccess={onProcessSuccess}
                  onProcessError={onProcessError}
                  rawData={rawData}
                  featureSet={featureSet}
                />             
              ) : (
                <p className="text-center mt-10">
                  Please select feature set and upload data first.
                </p>
              )
            }
          /> */}

          <Route
            path="/view_results"
            element={
              results ? (
                <div className="pt-20">
                  <ViewResults results={results} />
                </div>
              ) : modelList.length > 0 && rawData ? (
                // lazy-run models if possible
                <AutoRunResults
                  models={modelList}
                  rawData={rawData}
                  setResults={setResults}
                />
              ) : (
                <p className="text-center mt-10">
                  Please select models and upload data first.
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
        </div >
    </Router>
  );
}

// Helper component to auto-run models if /display_results is hit directly
function AutoRunResults({ models, rawData, setResults }) {
  const navigate = useNavigate();

  useEffect(() => {
    (async () => {
      try {
        const res = await ModelRun(models, rawData);
        setResults(res);
        navigate("/display_results"); // reload route with results
      } catch (err) {
        console.error("Error auto-running models:", err);
      }
    })();
  }, [models, rawData, setResults, navigate]);

  return <p className="text-center mt-10">Running models...</p>;
}

export default App
