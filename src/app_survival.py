# src/app_survival.py
import streamlit as st
import pandas as pd
import numpy as np
import requests, os, io, json, tempfile
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="SWAN Survival: Time to Menopause")

# ------- Settings from Streamlit Secrets -------
# Add these in Streamlit Cloud -> Settings -> Secrets
# GH_OWNER: your GitHub username or org (string)
# GH_REPO: private repo name with artifacts (e.g., "menopause-prediction")
# GH_TAG: release tag (e.g., "model-v1")
# GH_TOKEN: PAT with repo read access
owner  = st.secrets.get("GH_OWNER")
repo   = st.secrets.get("GH_REPO")
tag    = st.secrets.get("GH_TAG", "model-v1")
token  = st.secrets.get("GH_TOKEN")

if not all([owner, repo, token]):
    st.error("Missing GH_OWNER / GH_REPO / GH_TOKEN in secrets.")
    st.stop()

ASSET_NAMES = {
    "model": "cox_model.pkl",
    "meta": "model_meta.json",
    "cal": "calibration.joblib",
}

@st.cache_resource(show_spinner=False)
def download_release_assets(owner, repo, tag, token):
    """Download model artifacts from a private GitHub release into temp files and load them."""
    session = requests.Session()
    session.headers.update({"Authorization": f"token {token}",
                            "Accept": "application/vnd.github+json"})

    # Get release by tag
    rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    r = session.get(rel_url, timeout=30)
    r.raise_for_status()
    release = r.json()

    assets = {a["name"]: a for a in release.get("assets", [])}
    tmpdir = tempfile.mkdtemp(prefix="menopause_artifacts_")

    paths = {}
    for key, name in ASSET_NAMES.items():
        if name not in assets:
            raise FileNotFoundError(f"Asset '{name}' not found in release {tag}")
        asset = assets[name]
        # GitHub requires a special header to download binary asset
        dl = session.get(asset["url"], headers={"Accept": "application/octet-stream"}, timeout=120, stream=True)
        dl.raise_for_status()
        out_path = os.path.join(tmpdir, name)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        paths[key] = out_path

    # Load artifacts
    model = joblib.load(paths["model"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    calibrators = joblib.load(paths["cal"])
    return model, meta, calibrators

cph, meta, calibrators = download_release_assets(owner, repo, tag, token)

st.title("Time to Natural Menopause (Survival Model)")
st.write("Cox proportional hazards model trained on SWAN visits (V1–V10).")

status_label_to_ord = {
    "Pre-menopausal (regular cycles)": 2,
    "Early perimenopausal (cycle irregularity, <3m gap)": 1,
    "Late perimenopausal (>3m gap, <12m)": 0,
}

with st.sidebar:
    st.subheader("Baseline status")
    status_choice = st.selectbox(
        "Select your baseline menopausal status:",
        list(status_label_to_ord.keys()),
        index=1
    )
    q_label = st.selectbox("Event quantile", ["50% (median)", "25% events", "10% events"], index=0)
    q_map = {"50% (median)":0.5, "25% events":0.25, "10% events":0.10}
    q = q_map[q_label]
    st.caption("Model metrics: C-index (best) = {:.3f}".format(meta.get("c_index_best", float("nan"))))

def clip_by_meta(name, val):
    pct = meta.get("input_percentiles", {}).get(name)
    if not pct: return val
    p1, p99 = pct.get("p1"), pct.get("p99")
    if p1 is not None: val = max(val, p1)
    if p99 is not None: val = min(val, p99)
    return val

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", 35.0, 60.0, 46.0)
    bmi = st.number_input("BMI", 15.0, 60.0, 26.5)
with col2:
    fsh = st.number_input("FSH (mIU/mL)", 0.0, 300.0, 20.0)
    e2  = st.number_input("Estradiol (pg/mL)", 0.0, 400.0, 70.0)

age_c = clip_by_meta("age", float(age))
bmi_c = clip_by_meta("bmi", float(bmi))
fsh_c = clip_by_meta("fsh", float(fsh))
e2_c  = clip_by_meta("e2",  float(e2))

row = {
    "age": age_c,
    "bmi": bmi_c,
    "fsh_log": np.log1p(max(fsh_c, 0.0)),
    "e2_log":  np.log1p(max(e2_c, 0.0)),
    "fsh_e2_ratio": (fsh_c / (e2_c + 1e-6)) if e2_c>0 else float(fsh_c),
    "baseline_status_ord": status_label_to_ord[status_choice],
}
features = meta.get("features", ["age","bmi","fsh_log","e2_log","fsh_e2_ratio","baseline_status_ord"])
defaults = meta.get("defaults", {})
for f in features:
    if f not in row:
        row[f] = defaults.get(f, 0.0)
X = pd.DataFrame([row], columns=features)

horizons = {"1-year":12, "3-year":36, "5-year":60}
timeline = np.linspace(0, 360, 361)

try:
    surv = cph.predict_survival_function(X, times=timeline)
    s_vals = surv.iloc[:,0].values
    target_s = 1.0 - q
    idx = np.where(s_vals <= target_s)[0]
    t_q = float(timeline[idx[0]]) if len(idx)>0 else np.nan
    label_q = "median time" if q==0.5 else f"time to {int(q*100)}% events"
    if np.isfinite(t_q):
        st.metric(f"Predicted {label_q}", f"{t_q:.0f} months (~{t_q/12:.1f} years)")
    else:
        st.metric(f"Predicted {label_q}", "> 30 years")

    st.subheader("Risk within fixed horizons (calibrated)")
    cols = st.columns(len(horizons))
    for i,(label,m) in enumerate(horizons.items()):
        s_t = float(cph.predict_survival_function(X, times=[m]).iloc[0,0])
        p_raw = max(0.0, min(1.0, 1.0 - s_t))
        if str(m) in calibrators:
            p_cal = float(calibrators[str(m)].predict([p_raw])[0])
            p_disp = np.clip(p_cal, 0.0, 1.0)
            cols[i].metric(label, f"{p_disp*100:.1f}%", help=f"calibrated (raw: {p_raw*100:.1f}%)")
        else:
            cols[i].metric(label, f"{p_raw*100:.1f}%", help="raw")

    fig, ax = plt.subplots()
    ax.plot(timeline, s_vals)
    ax.set_xlabel("Months"); ax.set_ylabel("Probability not yet menopausal")
    ax.set_ylim(0,1); ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # Factor contributions (log-hazard)
    if st.checkbox("Show factor contributions (log-hazard)"):
        try:
            coefs = getattr(cph, "params_", None)
            if coefs is None:
                st.info("Model coefficients are not available; cannot compute contributions.")
            else:
                coefs = coefs.reindex(features)  # привести порядок к features
                xv = X.iloc[0].reindex(features).astype(float)
                contrib = coefs.values * xv.values
                contrib_df = pd.DataFrame({
                    "feature": features,
                    "coef": coefs.values,
                    "value": xv.values,
                    "contribution": contrib
                })
                # убрать NaN/Inf и отсортировать по величине вклада
                contrib_df = contrib_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["contribution"])
                contrib_df = contrib_df.sort_values("contribution", key=lambda s: s.abs(), ascending=False)
                st.write(contrib_df.head(8))
        except Exception as e:
            st.warning(f"Could not compute factor contributions: {e}")

    if st.checkbox("Download JSON report"):
        report = {
            "inputs": {"age":age_c, "bmi":bmi_c, "FSH":fsh_c, "E2":e2_c, "baseline_status":status_choice},
            "quantile": {"q": q, "label": label_q, "months": float(t_q) if np.isfinite(t_q) else None},
            "calibrated_risks": {
                str(m): (
                    float(calibrators[str(m)].predict([
                        max(0.0, min(1.0, 1.0 - float(cph.predict_survival_function(X, times=[m]).iloc[0,0])))
                    ])[0]) if str(m) in calibrators else None
                ) for m in horizons.values()
            },
            "model": {"features":features, "c_index_best": meta.get("c_index_best"),
                      "best_orientation": meta.get("best_orientation"),
                      "build_info": meta.get("build_info", {})}
        }
        st.download_button("Download JSON",
                           data=io.BytesIO(json.dumps(report, indent=2).encode("utf-8")),
                           file_name="menopause_prediction.json",
                           mime="application/json")

    with st.expander("About / Disclaimer"):
        bi = meta.get("build_info", {})
        st.write(f"Model features: {', '.join(features)}")
        st.write(f"C-index (best): {meta.get('c_index_best')}")
        st.write(f"Trained at (UTC): {bi.get('trained_at_utc', 'NA')}")
        st.caption("Research/educational prototype. Not medical advice.")

except Exception as e:
    st.warning(f"Could not compute survival curve: {e}")