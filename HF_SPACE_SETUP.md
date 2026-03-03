# Hugging Face Space Setup (Docker, no card)

## 1) Fix the variable error you saw

In your screenshot, the **Name** field is wrong.

- Wrong Name: `HF_MODEL_REPO_IDmerciless-admiral/200M_Param_GPT`
- Correct Name: `HF_MODEL_REPO_ID`
- Correct Value: `merciless-admiral/200M_Param_GPT`

For every variable:
- **Name** = only variable key
- **Value** = actual value

## 2) Space creation choices

- SDK: **Docker**
- Template: **Blank**
- Hardware: **CPU Basic (Free)**
- Visibility: **Public**

## 3) Push backend code to Space repo

After creating the Space, clone it and copy your backend files into it.

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cd <your-space-name>
# copy backend files here (Dockerfile, api.py, rag/, rag_index/, requirements.txt, etc.)
git add .
git commit -m "Deploy FastAPI backend"
git push
```

If Git asks for authentication, use your Hugging Face token as password.

## 4) Add Variables and Secrets

Go to **Space Settings → Variables and secrets** and add:

### Variables (public)
- `HF_MODEL_REPO_ID` = `merciless-admiral/200M_Param_GPT`
- `HF_MODEL_FILENAME` = `model_domain_tuned_new.pt`
- `ALLOWED_ORIGINS` = `https://<your-frontend-domain>`

### Secret (only if model repo is private)
- `HF_TOKEN` = `<your_hf_token>`

## 5) App startup

This repo now includes a Dockerfile that starts:

`uvicorn api:app --host 0.0.0.0 --port 7860`

No extra command needed in Space settings.

## 6) Check deployment

After build finishes:
- Open: `https://<your-username>-<your-space-name>.hf.space/api/health`
- Expect JSON with `status: healthy`

## 7) Connect frontend

In your frontend host (Vercel/Netlify), set:
- `NEXT_PUBLIC_API_URL` = `https://<your-username>-<your-space-name>.hf.space`

Then redeploy frontend.

## Notes

- Free Spaces may sleep when idle (normal on free tier).
- If you see CORS errors, verify `ALLOWED_ORIGINS` exactly matches frontend URL.
