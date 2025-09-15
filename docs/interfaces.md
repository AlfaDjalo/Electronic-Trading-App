# Interfaces Between Components

## Backend ↔ Frontend API

| Endpoint          | Method | Input (JSON)                           | Output (JSON)                   |
|-------------------|--------|----------------------------------------|---------------------------------|
| `/api/upload_data` | POST   | `{ fileName }`   | `{ rawData }`      |
| `/api/feature_sets` | GET   | `{ }`   | `{ featureSets }`      |
| `/api/functions` | GET   | `{ }`   | `{ functions }`      |
| `/api/save_feature_set` | POST   | `{ name, featureSet }`   | `{ }`      |
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

    class FunctionLoader {
        +load() functions
    }

    class FeatureSetSaver {
        +save( name, featureSet)
    }

    class ModelRunner {
      +run(processedData, modelList, hyperparameters) resultsData
    }

    class API {
      +POST /api/upload_data()
      +GET /api/load_feature_sets()
      +GET /api/functions()
      +POST /api/save_feature_set()
      +POST /api/run_models()
    }


    API --> DataLoader
    API --> FeatureSetLoader
    API --> FunctionLoader
    API --> FeatureSetSaver
    API --> ModelRunner