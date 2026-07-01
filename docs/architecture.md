# Architecture

```mermaid
flowchart TD
    A["SHL catalog JSON endpoint"] --> B["Scraper and cache"]
    B --> C["Normalized catalog dataset"]
    C --> D["Hybrid embeddings"]
    D --> E["FAISS index"]

    F["POST /chat messages"] --> G["Planner"]
    G -->|vague| H["Clarification reply"]
    G -->|refuse| I["Guardrail refusal"]
    G -->|compare| J["Catalog comparison"]
    G -->|recommend| K["Retrieval and reranking"]
    K --> L["Response composer"]
    J --> L
    H --> L
    I --> L
    L --> M["Exact JSON response"]
```

## Notes

- The service is stateless; every request carries the full transcript.
- The catalog JSON endpoint is the primary source of truth.
- The public website is a fallback path only if the endpoint is unavailable.
- Recommendations are limited to catalog items that look like individual assessments, not reports or packaged solutions.
