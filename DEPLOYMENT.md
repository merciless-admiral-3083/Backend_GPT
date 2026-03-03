# Backend Deployment Guide

This backend supports two model-loading modes:

1. **Local checkpoint** (set `MODEL_PATH` to a file in the container), or
2. **Hugging Face download** (recommended for cloud deploys).

## Required environment variables

- `ALLOWED_ORIGINS` (comma-separated, e.g. `https://your-frontend.com,http://localhost:3000`)
- `HF_MODEL_REPO_ID` (example: `merciless-admiral/200M_Param_GPT`)
- `HF_MODEL_FILENAME` (example: `model_domain_tuned_new.pt`)

## Optional environment variables

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN` (required only if your model repo is private)
- `HF_MODEL_REVISION` (branch/tag/commit in the model repo)
- `MODEL_PATH` (local path; defaults to `model_domain_tuned_new.pt`)

## Start command

The current Procfile is already valid:

`web: uvicorn api:app --host 0.0.0.0 --port $PORT`

## Example: Render/Railway deployment

1. Push this folder to GitHub.
2. Create a new Web Service from the repo.
3. Build command:

   `pip install -r requirements.txt`

4. Start command:

   `uvicorn api:app --host 0.0.0.0 --port $PORT`

5. Add environment variables listed above.
6. Deploy.

## Health check

Use:

`GET /api/health`

The response now reports whether GPT model loading succeeded (`model_loaded`) and the real RAG fact count.
