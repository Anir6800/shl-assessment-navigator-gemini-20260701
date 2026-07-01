# Approach

This solution is built around one principle: never let the model invent assessment data.

## Design

- Scrape the SHL catalog from the provided JSON endpoint.
- Normalize every record into a typed Pydantic model.
- Filter out obvious out-of-scope artifacts such as reports, guides, and packaged solutions.
- Build a FAISS index over recommendable assessments using a hybrid TF-IDF backend.
- Keep the API stateless and derive all context from the incoming transcript.

## Retrieval

The retriever combines:

- vector similarity from the FAISS index
- lexical overlap
- inferred category matches
- inferred job-level matches
- simple metadata boosts for remote/adaptive/duration preferences

This is intentionally lightweight so it can start quickly and still return strong top-10 recall for assessment queries.

## Prompts

Prompt text is stored in `app/prompts/` and loaded at runtime. The service uses deterministic fallback generation by default, but the prompt files are ready for an external Gemini, Claude, or OpenRouter provider.

## Evaluation

The included tests check:

- clarification on vague requests
- recommendation behavior
- refinement behavior
- comparison behavior
- refusal behavior
- schema compliance
- smoke-level recall at 10

## Tradeoffs

- I prioritized reliability and speed over a heavier LLM-only pipeline.
- The current embedding backend defaults to a local hybrid model, which is easier to ship than a remote embedding API.
- The system can still swap in BGE-style embeddings or a hosted LLM later without changing the API contract.

## Limitations

- Comparisons are grounded but template-based.
- The local embedding backend is lexical-heavy, so a hosted semantic embedding model may improve recall on harder traces.

## Future improvements

- Add a stronger semantic embedding backend once the environment is stable.
- Introduce trace-driven prompt tuning against the hidden evaluation set.
- Add more aggressive alias expansion for assessment acronyms and abbreviations.
