# Interfaces Between Components

## Backend ↔ Frontend API

| Endpoint          | Method | Input (JSON)                           | Output (JSON)                   |
|-------------------|--------|----------------------------------------|---------------------------------|
| `/api/load_data` | POST   | `{ fileName }`   | `{ rawData }`      |
| `/api/feature_sets` | POST   | `{ }`   | `{ featureSets }`      |
| `/api/run_models` | POST   | `{ processedData, modelList, hyperparameters }`   | `{ resultsData }`      |


## Internal Component Interfaces

```mermaid
classDiagram
    class DataLoader {
      +load(fileName) rawData
    }

    class FeatureSetLoader {
      +load() featureSets
    }

    class ModelRunner {
      +run(processedData, modelList, hyperparameters) resultsData
    }

    class API {
      +POST /api/load_data()
      +POST /api/load_feature_sets()
      +POST /api/run_models()
    }


    API --> DataLoader
    API --> FeatureSetLoader
    API --> ModelRunner