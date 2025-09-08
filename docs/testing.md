```mermaid
graph TD
    A[Testing Strategy] --> B[Backend]
    A --> C[Frontend]
    A --> D[End-to-End]

    B --> B1[Unit Tests]
    B --> B2[Integration Tests]
    B --> B3[Performance Tests]

    C --> C1[Component Tests]
    C --> C2[Integration/UI Tests]

    D --> D1[User Journeys]