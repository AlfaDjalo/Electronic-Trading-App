# Interfaces Between Components

## Backend ↔ Frontend API

| Endpoint          | Method | Input (JSON)                           | Output (JSON)                   |
|-------------------|--------|----------------------------------------|---------------------------------|
| `/api/load_data` | POST   | `{ fileName }`   | `{ rawData }`      |
| `/api/process_data` | POST   | `{ rawData, featureSet }`   | `{ processedData }`      |
| `/api/run_models` | POST   | `{ processedData, modelList, hyperparameters }`   | `{ resultsData }`      |


## Internal Component Interfaces

```mermaid
classDiagram
    class DataLoader {
      +load(fileName) rawData
    }

    class DataProcessor {
      +process(rawData, featureSet) processedData
    }

    class ModelRunner {
      +run(processedData, modelList, hyperparameters) resultsData
    }

    class API {
      +POST /api/load_data()
      +POST /api/process_data()
      +POST /api/run_models()
    }


    API --> DataLoader
    API --> DataProcessor
    API --> ModelRunner