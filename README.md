# menopause-prediction-site

Public Streamlit UI for the Time‑to‑Menopause survival model.  
Model artifacts are fetched at runtime from a private GitHub release using a token stored in Streamlit Secrets.

## Local run (requires access to the private repo release)
Set environment variables for testing:
- `GH_OWNER`, `GH_REPO`, `GH_TAG` (e.g., `model-v1`), `GH_TOKEN` (PAT with repo read)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GH_OWNER=your_user
export GH_REPO=menopause-prediction
export GH_TAG=model-v1
export GH_TOKEN=<your_PAT>
streamlit run src/app_survival.py