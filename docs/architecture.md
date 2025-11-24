# System Architecture

The project consists of:
- **Backend** (Python) -> handles API, data processing, ML models
- **Frontend** (React, Vite, Tailwind) -> handles UI, charting, feature selection
- **Deployment** static frontend hosted on GitHub pages, backend hosted separately (render ?)

```mermaid
flowchart LR
    subgraph Frontend [React Frontend]
        A[DataUpload.jsx]
        B[ViewData.jsx]
        C[FeatureSetManager.jsx]
        D[ModelSelect.jsx]
        E[ViewResults.jsx]
    end

    subgraph Backend [Python]
        F[/api/run_models/]
        G[Data Models]
    end

    A <-->|Reads in CSV file| F
    C <-->|Reads in JSON file| F
    D <-->|Runs keras models| F
    F <--> G

