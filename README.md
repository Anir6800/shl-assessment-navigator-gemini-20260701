# SHL Assessment Recommender

Conversational FastAPI service for recommending SHL Individual Test Solutions from a scraped catalog.

The app also serves a ChatGPT-style browser UI at `/` and includes a generated PDF implementation report in `outputs/`.

## What it does

- Clarifies vague assessment requests.
- Recommends 1 to 10 grounded assessments from the catalog.
- Refines shortlists when the user adds constraints.
- Compares two assessments using catalog data only.
- Refuses prompt injection, general hiring advice, and legal advice.

## API

- `GET /` -> ChatGPT-style single-page UI
- `GET /health` -> `{"status":"ok"}`
- `POST /chat`

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment."}
  ]
}
```

Response body:

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": true
}
```

## Local setup

```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` in a browser after starting the server.

## Environment variables

- `SHL_EMBEDDING_BACKEND` - `auto` or `hybrid_tfidf`
- `SHL_LLM_PROVIDER` - `auto`, `gemini`, `anthropic`, or `openrouter`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `SHL_MAX_RECOMMENDATIONS`
- `SHL_RETRIEVAL_CANDIDATE_POOL`
- `SHL_CACHE_TTL_HOURS`
- `SHL_VECTOR_STORE_DIR`

## Docker and Gemini

If you want to run the app with Gemini in Docker:

```bash
cp .env.example .env
docker compose up --build
```

Set `GEMINI_API_KEY` in `.env` and keep `SHL_LLM_PROVIDER=gemini` if you want Gemini to polish replies. The app falls back to a deterministic template if the key is absent.

## Deployment

- Docker: build with the included `Dockerfile` and expose port `7860` or the `PORT` env var.
- Render: use [`render.yaml`](./render.yaml).
- Railway: use [`railway.toml`](./railway.toml) or the `Procfile`.
- Hugging Face Spaces: deploy as a Docker Space using the same `Dockerfile`.

## Testing

```bash
pytest -q
python tests/evaluate_smoke.py
```

## Docs

- [Architecture](./docs/architecture.md)
- [Approach](./docs/approach.md)

## PDF report

The generated report is written to `outputs/shl_assessment_recommender_report.pdf`.
