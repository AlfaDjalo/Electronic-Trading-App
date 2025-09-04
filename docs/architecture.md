# System Architecture

The project consists of:
- **Backend** (Python) -> handles API, data processing, ML models
- **Frontend** (React, Vite, Tailwind) -> handles UI, charting, feature selection
- **Deployment** static frontend hosted on GitHub pages, backend hosted separately

```mermaid
flowchart LR
    subgraph Frontend [React Frontend]
        A[ChartArea.jsx]
        B[ModelSelect.jsx]
    end

    subgraph Backend [Python]
        C[/api/run_models/]
        D[Data Models]
    end

    A -->|Fetch JSON| C
    B -->|Sends user input| C
    C --> D

