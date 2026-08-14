# Parallel Model Test -- setup notes

New page added to the existing dashboard at `app/dashboard.py` -- open the app as
usual and pick "Parallel Model Test" from the sidebar. Everything below is
about getting it running for the first time.

## Install

```
pip install -r requirements.txt
```

New dependencies pulled in for this feature: `litellm` (the BYOK layer that
calls any provider through one interface) and `requests` (for the "datasheet
by URL" input). Nothing else changed.

## Run

Same as before:

```
streamlit run app/dashboard.py
```

## What's new

Three tabs live under "Parallel Model Test":

- **Model Registry** -- add any model by pasting its API key. Pick a known
  model from the dropdown (pre-fills provider/endpoint/vision flag) or choose
  "Other / custom model" to point at anything else. No code touched, ever --
  it writes to a local SQLite file at `model_testing_data/model_testing.db`
  (gitignored, created automatically on first use).
- **Run Test** -- upload a datasheet PDF or paste a URL to one, pick a chunk,
  check off which registered models to run, optionally pick a judge model for
  quality scoring, and hit run. All selected models are called in parallel via
  `asyncio` + LiteLLM. Tokens, cost, latency and (if a judge was picked) a
  0-100 quality score come back per model.
- **Leaderboard** -- every run anyone has ever kicked off, persisted and
  aggregated per model. Ranked by quality first, then cost. Includes a
  cost-vs-quality scatter and the full raw run history.

## On quality scoring

The call site being tested is `src/ingestion/enrich.py`'s
`create_ai_enhanced_summary` -- it asks a model for a free-text *searchable
summary* of a chunk, not a fixed extraction schema. So scoring here is an
LLM-as-judge rubric graded against the original chunk content (facts
preserved, topic coverage, searchability, no fabrication), not a
Pydantic/JSON schema match. Pick a judge model that isn't also one you're
testing so nothing grades its own homework. Quality scoring is optional --
leave the judge dropdown on "No quality scoring" to just see tokens/cost/
latency.

## Data note

This delivery excludes the repo's existing sample exports and vector DB
(`dbv1/`, `dbv2/`, `json/`, `parsed_pdf/`, the root-level `chunks_*.json` /
`rag_results.json`) since those are large generated artifacts already sitting
in the GitHub repo and weren't needed to ship the new feature. Everything
else -- source, sample datasheet PDFs in `data/`, the API server, tests -- is
here. Pull those directories from the GitHub repo directly if you want the
prior ingestion results locally too.
