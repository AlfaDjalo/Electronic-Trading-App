# Backend API Flow

This page documents the high-level API flows. Diagrams are rendered using Mermaid.

## Endpoints covered

- `POST /api/upload_data` — accepts CSV uploads and returns time series JSON
- `POST /api/yahoo_data` — fetches historical data from Yahoo Finance
- `GET /api/feature_sets` — returns available feature set definitions
- `POST /api/run_models` — primary business endpoint: runs models and returns predictions and statistics

## API flow (overview)

```mermaid
flowchart LR
  subgraph Frontend
    UI[User Interface]
  end

  UI -->|"upload / choose data"| API["/api/upload_data or /api/yahoo_data"]
  API --> Backend[api_routes.py]
  Backend -->|load feature_sets| FS["file: back-end/data/feature_sets.json"]
  Backend -->|call| Service[process_models_request]
  Service --> Runner["training/runner.py::ModelRunner"]
  Runner -->|prepare| DataProc[process_data.py::DataProcessor]
  Runner -->|train| Trainer["training/trainer.py::ModelTrainer"]
  Trainer -->|create| Factory["models/factory.py::ModelFactory"]
  Factory --> Implementations["models/implementations.py"]
  Trainer -->|evaluate| Metrics[sklearn metrics]
  Runner -->|aggregate| Results[Predictions + Stats]
  Results --> Backend
  Backend -->|json response| UI

  classDef backend fill:#cce5ff,stroke:#003366,stroke-width:1px;
  classDef backend fill:#cce5ff,stroke:#003366,color:#000,stroke-width:1px;
  class Backend,Service,Runner,DataProc,Trainer,Factory backend
```

## Notes
- The flow shows the logical responsibilities — details (caching, grouping) are shown in the model running doc.
- `process_models_request` (in `services/model_service.py`) is the business-logic bridge used by the API to invoke the runner and collect results.
