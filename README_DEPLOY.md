
# Deployment instructions (Hugging Face Spaces and Render)

This document explains two deployment options: **Hugging Face Spaces (easiest for demos)** and **Render** (more flexible, production-ready). It also includes required files in this repo.

## Included new files
- `requirements.txt` - Python dependencies
- `app.py` - Gradio app wrapper to serve a simple web UI
- `Dockerfile` - For deploying to Render or any Docker-based host
- This `README_DEPLOY.md` - this file

> Note: The original `kirtpro3.py` in the repo is left unchanged. `app.py` will try to import `generate_image(prompt)` from it. If not available or it errors, the app shows a fallback placeholder image.

---

## Option A — Hugging Face Spaces (recommended for quick demos)
1. Create a Hugging Face account and verify email.
2. Create a new **Space** -> choose `Gradio` as the SDK.
3. Initialize the Space git repo (you can use their web upload or `git`).
4. Push the contents of this project (all files) to the Space repo.
   - Important files: `app.py`, `requirements.txt`, `kirtpro3.py`, any model/artifacts.
5. If your project requires Hugging Face model weights:
   - Add `HF_HUB_TOKEN` as a secret in the Space settings (if the model is private).
   - In `kirtpro3.py`, you can load a model with `from diffusers import DiffusionPipeline` and `pipeline = DiffusionPipeline.from_pretrained("model-id", use_auth_token=os.environ.get("HF_HUB_TOKEN"))`
6. Spaces will automatically build and run. Visit the Space URL.

Pros: easiest, free tier for demos, automatic build.  
Cons: limited compute (no GPUs on free), less control.

---

## Option B — Render (recommended for custom backend / production)
Two ways: Deploy using Docker (recommended) or as a Python web service.

### Deploy with Docker (steps)
1. Create a Render account.
2. Create a new **Service** -> choose "Web Service".
3. Connect your GitHub repo or upload repo.
4. For Docker:
   - Render will detect the `Dockerfile` and build image.
   - Set environment variables (e.g., `PORT=7860`, `HF_HUB_TOKEN`).
5. If your model needs large memory/GPU, consider Render's GPU instances (paid).
6. Deploy and open the service URL.

### Deploy without Docker (Python)
1. Create a new Web Service on Render, select Python.
2. Ensure `requirements.txt` lists all dependencies.
3. Set the start command to `python app.py` and set `PORT` env var if needed.

Pros: more control, custom instance sizes, ability to use GPUs (paid).  
Cons: slightly more setup than Spaces.

---

## Files to update/add if you have model weights
- If you have Hugging Face model id or local checkpoint, modify `kirtpro3.py` to implement:
```py
def generate_image(prompt: str):
    # load pipeline (cache model load globally)
    # produce PIL Image and return it or path to saved image
    return pil_image
```
- If using a pretrained model from Hugging Face Hub:
  - Set env var `HF_HUB_TOKEN` in deployment platform and use `from_pretrained(model_id, use_auth_token=...)`

---

## Quick local testing
1. Create a virtual env: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`
3. Run: `python app.py` and open `http://localhost:7860`

---

## Troubleshooting / Tips
- If your model requires GPU and you only have CPU, inference will be slow or impossible.
- For large models, prefer loading with `torch_dtype="auto"` and `device_map="auto"` when using `accelerate`.
- Add a small health check endpoint if deploying behind load balancers.
