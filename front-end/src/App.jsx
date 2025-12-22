import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import './App.css'
import "./index.css"

import { Navbar } from "./components/Navbar"
import { MobileMenu } from "./components/MobileMenu"
import { DataUpload } from './components/DataUpload';
// import { DataSourceSelector } from './components/DataSourceSelector';
import { ViewData } from "./components/ViewData";
import { ViewResults } from "./components/ViewResults";
import { ModelSelect } from "./components/ModelSelect";
import { FeatureSetManager } from "./components/FeatureSetManager";
import { useModelConfig } from "./hooks/useModelConfig";
import { BacktestSelect } from "./components/BacktestSelect";
import BacktestResults from "./components/BacktestResults";
// import { ProcessData } from "./components/ProcessData";
import { ModelRun } from "./components/ModelRun";
// import { DisplayResults } from "./components/DisplayResults";
// import featureSetFile from "./feature_sets.json";
import FeatureSetLoader from "./components/FeatureSetLoader";

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
  const [availableFeatureSets, setAvailableFeatureSets] = useState([]);
  const [allFeatureSets, setAllFeatureSets] = useState([]);
  const [backtestList, setBacktestList] = useState([]);
  const [backtestResults, setBacktestResults] = useState(null);
  // const [availableFeatureSets, setAvailableFeatureSets] = useState({});
  // const [allFeatureSets, setAllFeatureSets] = useState({});
  const [functionList, setFunctionList] = useState([])
  const basename =
    import.meta.env.DEV ? "/" : "/electronic-trading-app";
  // const modelNames = ["Baseline", "LSTM", "CNN", "MLP"]
  // const featureSets = ["Limit_Order_Book", "Cut_Down_Limit_Order_Book", "Sandbox_Intraday"]
  // const [selectedFeatures, setSelectedFeatures] = useState([]);
  
  const addModel = (newModel) => {
    // Build base name (without number)
    const baseName = `${newModel.model}-${newModel.featureSet}`;

    // Count existing models with this base name
    const count = modelList.filter(m => 
      m.name && m.name.startsWith(baseName)
    ).length;

    // Create numbered name
    const numberedName = `${baseName}_${count + 1}`;

    // Assign the numbered name
    const modelWithNumberedName = { ...newModel, name: numberedName };

    setModelList((prev) => [...prev, modelWithNumberedName]);
  };

  // const addModel = (newModel) => setModelList((prev) => [...prev, newModel])
  const deleteModel = (id) => setModelList((prev) => prev.filter((m) => m.id !== id));
  const updateModel = (updated) =>
    setModelList((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));
  // const editModel = (model) => {
    //   console.log("Edit clicked:", model);
  // };
  const setParameters = (model) => {
    console.log("Set parameters for:", model);
  };
  
  const addBacktest = (newBT) => {
    // Build base name (without number)
    const baseName = `${newBT.model}`;

    // Count existing models with this base name
    const count = backtestList.filter(m => 
      m.name && m.name.startsWith(baseName)
    ).length;

    // Create numbered name
    const numberedName = `${baseName}_${count + 1}`;

    // Assign the numbered name
    const backtestWithNumberedName = { ...newBT, name: numberedName };

    setBacktestList((prev) => [...prev, backtestWithNumberedName]);
  };

  // const addModel = (newModel) => setModelList((prev) => [...prev, newModel])
  const deleteBacktest = (id) => setBacktestList((prev) => prev.filter((m) => m.id !== id));
  const updateBacktest = (updated) =>
    setBacktestList((prev) => prev.map((m) => (m.id === updated.id ? updated : m)));

  useEffect(() => {
    // fetch list of backend-supported functions
    fetch("http://localhost:5000/api/functions")
    .then((res) => res.json())
    .then((data) => setFunctionList(data))
    .catch((err) => console.error("Failed to load functions", err));
  }, []);
  
  const handleDataLoad = (json) => {
    if (json.success) {
      setRawData(json.timeSeriesData);

      // If backend also provides feature sets for this dataset
      if (json.featureSets) {
        setAllFeatureSets(json.featureSets);
        // setFeatureSets(json.featureSets);
      }
    } else {
      console.error("Failed to load data:", json.error);
    }
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
      
      const filtered = Object.entries(allFeatureSets || {})
        .filter(([fsName, fs]) => {
          if (!fs || !Array.isArray(fs.features)) return false;

          return fs.features.every(feature => {
            const fields = feature.input_data_fields || feature.inputDataFields || [];
            return fields.every(f => names.includes(f));
          });
        })
        .map(([name, fs]) => ({ name, ...fs }));

      setAvailableFeatureSets(filtered);
      console.log(filtered);
    } else {
      console.warn("No rows found in upload results");
      setSeriesNames([]);
      setAvailableFeatureSets([]);
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
      
      const modelConfig = useModelConfig();
      
      if (!modelConfig) return <p>Loading model config...</p>;
      
      const modelNames = Object.keys(modelConfig);

      return (
        // <Router>
        <Router basename={basename}>  
        {/* Always visible */}
        <Navbar menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>
        <MobileMenu menuOpen={menuOpen} setMenuOpen={setMenuOpen}/>

        <div className="pt-16">
          <Routes>

            <Route
              path="/"
              element={
                <div>
                  <h2> Electronic Trading App</h2>
                </div>
              }
            />

            {/* Data Upload */}
            <Route
              path="/load_data"
              element={
                <div>
                  {/* <DataSourceSelector onUploadSuccess={handleDataLoad} /> */}
                  <DataUpload 
                    onUploadSuccess={(results, fileName) =>
                      handleUploadSuccess(results, fileName)
                    }
                    onUploadError={handleUploadError}
                    uploadedFileName={uploadedFileName}
                  />
                  {/* Load feature sets once and store them */}
                  <FeatureSetLoader onLoaded={setAllFeatureSets} />

                  {/* <h3>Filtered Feature Sets</h3>
                  <pre>{JSON.stringify(availableFeatureSets, null, 2)}</pre> */}
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

            <Route
              path="/select_model"
              element={
                rawData ? (
                  <div className="pt-20">   {/* <-- offset for navbar height */}
                    <ModelSelect
                      modelNames={modelNames}
                      modelConfig={modelConfig}
                      featureSets={availableFeatureSets}
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

          <Route
            path="/manage_feature_sets"
            element={
              rawData ? (
                <FeatureSetManager 
                  availableFeatureSets={availableFeatureSets}
                  seriesNames={seriesNames}
                  functionList={functionList}
                />
              ) : (
                <p className="text-center mt-10">
                  Please upload data first.
                </p>
              )
            }
          />

          <Route
            path="/view_results"
            element={
              results && rawData ? (
                <div className="pt-20">
                  <ViewResults results={results} rawData={rawData} />
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

          <Route
            path="/run_backtest"
            element={
              results && rawData ? (
                <div className="pt-20">
                <BacktestSelect 
                  modelList={modelList}
                  backtestList={backtestList}
                  results={results}
                  onAddBacktest={addBacktest}
                  onUpdateBacktest={updateBacktest}
                  onDeleteBacktest={deleteBacktest}
                  setBacktestResults={setBacktestResults}
                />
                </div>
              ) : modelList.length > 0 && rawData ? (
                // lazy-run models if possible
                <BacktestSelect 
                  modelList={modelList}
                  backtestList={backtestList}
                  results={results}
                  onAddBacktest={addBacktest}
                  onUpdateBacktest={updateBacktest}
                  onDeleteBacktest={deleteBacktest}                  
                />
              ) : (
                <p className="text-center mt-10">
                  Please select models and upload data first.
                </p>
              )
            }
          />

          <Route
            path="/view_backtests"
            element={
              backtestResults && rawData ? (
                <div className="pt-20">
                  <BacktestResults backtestResults={backtestResults} />
                </div>
              ) : (
                <p className="text-center mt-10">Please run backtests to view results.</p>
              )
            }
          />
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
