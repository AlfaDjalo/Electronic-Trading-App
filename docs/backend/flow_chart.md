# Program Flow: DataProcessor & WindowGenerator

This document describes the flow of data processing in the electronic trading app backend.

## Flow Overview

```mermaid
flowchart TD
    A[Raw Data Input] --> B[DataProcessor]
    B --> C[Feature Engineering]
    C --> D[Data Normalization]
    D --> E[Split Data: Train / Val / Test]
    E --> F[WindowGenerator Initialization]
    F --> G[Generate Sliding Windows]
    G --> H[Train Window, Val Window, Test Window]
    H --> I[Ready for Model Training]

    %% Optional details
    B -->|Uses feature_set| C
    F -->|Requires input_width, label_width, shift| G
