# src/app_survival.py
# Public Streamlit UI that downloads model artifacts from a private GitHub release.
# UX:
# - Left column: "How to use" with instructions and parameter definitions
# - Right column: inputs (status, age int years, smoking, FSH/E2, height/weight with BMI auto-calc)
# - Big "Predicted median time", survival plot with 1/3/5-year markers
# - Expanders: Detailed Prognosis, Recommendations, Technical Details
# - JSON report download
#
# NOTE: This tool is educational only. Not medical advice.

import streamlit as st
import pandas as pd
import numpy as np
import requests, os, io, json, tempfile
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="My Menopause Forecast", layout="wide")

# Minimal theming for larger headline/metrics
st.markdown("""
<style>
.big-metric { font-size: 2.0rem; font-weight: 700; margin: 0.25rem 0 1rem 0; }
.side-note { color: #d14; font-size: 0.95rem; }
h1, h2, h3 { font-weight: 700; }
.block-label { font-weight: 600; }
.small { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ------- Settings from Streamlit Secrets -------
# Add these in Streamlit Cloud -> Settings -> Secrets:
# GH_OWNER, GH_REPO, GH_TAG (e.g., "model-v1"), GH_TOKEN (repo read)
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
def download_release_assets(owner: str, repo: str, tag: str, token: str):
    """Download model artifacts from a private GitHub release into temp files and load them."""
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    })

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
        dl = session.get(asset["url"], headers={"Accept": "application/octet-stream"},
                         timeout=120, stream=True)
        dl.raise_for_status()
        out_path = os.path.join(tmpdir, name)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        paths[key] = out_path

    model = joblib.load(paths["model"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    calibrators = joblib.load(paths["cal"])
    return model, meta, calibrators

# Load artifacts
cph, meta, calibrators = download_release_assets(owner, repo, tag, token)

st.title("My Menopause Forecast")

# ----- Left: How to use / Right: Inputs + Output -----
left, right = st.columns([1.1, 1.9])

with left:
    st.header("How to use:")
    st.markdown("""
This tool does NOT provide medical advice. Use it for educational purposes only.
It is a statistical estimate, not a guarantee. Please consult your doctor with any personal health questions.
This tool is intended for users who DO NOT have a health condition (other than perimenopause itself) that affects menstrual regularity.

1) Enter your health parameters in the boxes on the right.
2) The model will generate your personalized forecast below.
3) You will see a "Predicted median time" and a graph showing how the probability of entering menopause changes over time.
    """)
    st.markdown("**PROVIDED HEALTH PARAMETERS:**")
    with st.expander("Menopausal status"):
        st.markdown("""
- Pre-menopausal: Had bleeding in the last 3 months and periods "stayed the same".
- Early perimenopause: Had bleeding in the last 3 months, but periods had become "farther apart, closer together, more variable, or more regular".
- Late perimenopause: Had bleeding in the last 12 months, but no bleeding in the last 3 months.
        """)
    with st.expander("Age"):
        st.markdown("- Age between 35 and 60 in full years (no decimals).")
    with st.expander("Height and Weight"):
        st.markdown("- Height between ~140 and 200 cm.\n- Weight between ~40 and 160 kg.\n- BMI is auto-calculated; you may override it.")
    with st.expander("FSH and Estradiol (E2)"):
        st.markdown("""
Please provide values measured on days 2–5 of your cycle if your cycle is regular (premenopause), or from any day if your cycle is irregular (perimenopause).
- FSH typical range in the tool: 0–300 mIU/mL.
- E2 typical range in the tool: 0–400 pg/mL.
        """)

# Helper: clip raw numeric inputs by training percentiles
def clip_by_meta(name: str, val: float) -> float:
    pct = meta.get("input_percentiles", {}).get(name)
    if not pct:
        return val
    p1, p99 = pct.get("p1"), pct.get("p99")
    if p1 is not None:
        val = max(val, p1)
    if p99 is not None:
        val = min(val, p99)
    return val

status_label_to_ord = {
    "Pre-menopausal (regular cycles)": 2,
    "Early perimenopausal (cycle irregularity, <3m gap)": 1,
    "Late perimenopausal (>3m gap, <12m)": 0,
}

with right:
    # Inputs layout (two columns)
    c1, c2 = st.columns(2)

    with c1:
        status_choice = st.selectbox(
            "Menopausal Status",
            list(status_label_to_ord.keys()),
            index=1
        )
        # Age: integer years only
        age = st.number_input("Age (years)", min_value=35, max_value=60, value=46, step=1, format="%d")
        smoke_choice = st.selectbox("Currently smoking?", ["No", "Yes"], index=0)
    with c2:
        fsh = st.number_input("FSH (mIU/mL)", min_value=0.0, max_value=300.0, value=20.0, step=0.1)
        e2  = st.number_input("Estradiol (pg/mL)", min_value=0.0, max_value=400.0, value=70.0, step=0.1)

    # Height/Weight with BMI auto-calc + optional manual override
    st.markdown("**Anthropometrics**")
    c3, c4, c5 = st.columns([1, 1, 1])
    with c3:
        height_cm = st.number_input("Height (cm)", min_value=140.0, max_value=200.0, value=164.0, step=0.5)
    with c4:
        weight_kg = st.number_input("Weight (kg)", min_value=40.0, max_value=160.0, value=62.0, step=0.5)
    with c5:
        manual_bmi = st.checkbox("Override BMI", value=False)

    # Compute BMI or use manual override
    if manual_bmi:
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=26.5, step=0.1, help="Manual override")
    else:
        h_m = max(0.5, float(height_cm) / 100.0)
        bmi = float(weight_kg) / (h_m * h_m)
        st.metric("Calculated BMI (kg/m²)", f"{bmi:.1f}")

    # Clip inputs to training percentiles
    age_c = clip_by_meta("age", float(age))
    bmi_c = clip_by_meta("bmi", float(bmi))
    fsh_c = clip_by_meta("fsh", float(fsh))
    e2_c  = clip_by_meta("e2",  float(e2))
    smokere_bin = 1.0 if smoke_choice == "Yes" else 0.0

    # Compute FSH/E2 ratio consistent with training, optional p99 clipping
    ratio = float(fsh_c / (e2_c + 1e-6))
    r_p99 = meta.get("ratio_p99", None)  # may be absent; OK
    if r_p99 is not None:
        ratio = min(ratio, r_p99)

    # Build model input row
    features = meta.get("features", ["age", "bmi", "fsh_log", "e2_log", "fsh_e2_ratio", "baseline_status_ord"])
    defaults = meta.get("defaults", {})
    row = {
        "age": age_c,
        "bmi": bmi_c,
        "fsh_log": np.log1p(max(fsh_c, 0.0)),
        "e2_log":  np.log1p(max(e2_c, 0.0)),
        "fsh_e2_ratio": ratio,
        "baseline_status_ord": status_label_to_ord[status_choice],
        "smokere_bin": smokere_bin,  # will be used if present in `features`
    }
    for f in features:
        if f not in row:
            row[f] = defaults.get(f, 0.0)
    X = pd.DataFrame([row], columns=features)

    # Timeline and horizons
    horizons = {"1-year": 12, "3-year": 36, "5-year": 60}
    timeline = np.linspace(0, 360, 361)  # months up to 30 years

    # Main inference
    try:
        surv = cph.predict_survival_function(X, times=timeline)
        s_vals = surv.iloc[:, 0].values

        # Predicted median time (S(t)=0.5)
        idx = np.where(s_vals <= 0.5)[0]
        t_median = float(timeline[idx[0]]) if len(idx) > 0 else np.nan

        # Big metric
        if np.isfinite(t_median):
            st.markdown(f'<div class="big-metric">Predicted median time: {t_median:.0f} months (~{t_median/12:.1f} years)</div>', unsafe_allow_html=True)
            st.markdown('<div class="side-note">By this time, 50% of women with identical parameters are predicted to have entered menopause.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="big-metric">Predicted median time: > 30 years</div>', unsafe_allow_html=True)

        # Detailed prognosis (expander)
        with st.expander("Detailed Prognosis"):
            cols = st.columns(len(horizons))
            risk_cache = {}
            for i, (label, m) in enumerate(horizons.items()):
                s_t = float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])
                p_raw = float(np.clip(1.0 - s_t, 0.0, 1.0))
                if str(m) in calibrators:
                    p_cal = float(np.clip(calibrators[str(m)].predict([p_raw])[0], 0.0, 1.0))
                    cols[i].metric(label, f"{p_cal*100:.1f}%", help=f"calibrated (raw: {p_raw*100:.1f}%)")
                else:
                    p_cal = None
                    cols[i].metric(label, f"{p_raw*100:.1f}%", help="raw")
                risk_cache[m] = (p_raw, p_cal)

            # Survival curve with 1/3/5-year markers
            fig, ax = plt.subplots()
            ax.plot(timeline, s_vals, color="#2b6cb0", linewidth=2.0, label="Survival S(t)")
            for m in horizons.values():
                ax.axvline(m, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
                s_t = float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])
                p_evt = 1.0 - s_t
                ax.annotate(f"{p_evt*100:.1f}%", xy=(m, max(0.05, s_t)), xytext=(m+4, min(0.95, s_t+0.15)),
                            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6), fontsize=9, color="gray")
            ax.set_xlabel("Months")
            ax.set_ylabel("Probability not yet menopausal (S)")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            st.pyplot(fig)

            # Optional extra quantiles
            q_cols = st.columns(2)
            for q_label, q in [("25% events", 0.25), ("10% events", 0.10)]:
                target_s = 1.0 - q
                idxq = np.where(s_vals <= target_s)[0]
                t_q = float(timeline[idxq[0]]) if len(idxq) > 0 else np.nan
                if np.isfinite(t_q):
                    q_cols[0 if q==0.25 else 1].metric(f"Time to {q_label}", f"{t_q:.0f} months (~{t_q/12:.1f} years)")
                else:
                    q_cols[0 if q==0.25 else 1].metric(f"Time to {q_label}", "> 30 years")

        # Recommendations (education-only)
        with st.expander("Recommendations"):
            recs = []
            if smokere_bin >= 0.5:
                recs.append({
                    "title": "Smoking",
                    "text": "Smoking is associated in observational studies with earlier natural menopause. Quitting improves overall health. Consider discussing cessation strategies with a clinician.",
                    "links": [
                        "https://www.cdc.gov/tobacco/index.htm",
                        "https://www.who.int/health-topics/tobacco"
                    ]
                })
            if bmi_c < 18.5:
                recs.append({
                    "title": "Body weight",
                    "text": "Very low BMI can be associated with hormonal irregularities. Balanced nutrition and adequate energy intake support overall reproductive health.",
                    "links": [
                        "https://www.womenshealth.gov/menopause/menopause-symptoms-and-relief"
                    ]
                })
            if bmi_c >= 30:
                recs.append({
                    "title": "Lifestyle",
                    "text": "Healthy lifestyle (balanced diet, physical activity, sleep, stress management) supports overall cardiovascular and metabolic health.",
                    "links": [
                        "https://www.who.int/news-room/fact-sheets/detail/healthy-diet"
                    ]
                })
            recs.append({
                "title": "Repeat labs",
                "text": "FSH and estradiol levels can vary across cycles and labs. If values look unusual or borderline, consider repeating tests after several weeks/months for stability.",
                "links": []
            })
            if not recs:
                st.caption("No specific suggestions based on current inputs.")
            else:
                for r in recs:
                    links_block = "\n".join([f"- {u}" for u in r.get("links", [])]) if r.get("links") else ""
                    st.info(f"• {r['title']}: {r['text']}\n{links_block}")
            st.caption("These are educational notes only and not medical advice.")

        # Technical Details
        with st.expander("Technical Details"):
            st.markdown("Cox proportional hazards model trained on SWAN visits (V1–V10).")
            st.markdown(f"Model features: {', '.join(features)}")
            st.markdown(f"C-index (best): {meta.get('c_index_best')}")
            bi = meta.get("build_info", {})
            st.markdown(f"Trained at (UTC): {bi.get('trained_at_utc', 'NA')}")
            # Factor contributions
            if st.checkbox("Show factor contribution table (log-hazard)"):
                try:
                    coefs = getattr(cph, "params_", None)
                    if coefs is None:
                        st.info("Model coefficients are not available.")
                    else:
                        coefs = coefs.reindex(features)
                        xv = X.iloc[0].reindex(features).astype(float)
                        contrib = coefs.values * xv.values
                        contrib_df = pd.DataFrame({
                            "feature": features,
                            "coef": coefs.values,
                            "value": xv.values,
                            "contribution": contrib
                        }).replace([np.inf, -np.inf], np.nan).dropna(subset=["contribution"])
                        contrib_df = contrib_df.sort_values("contribution", key=lambda s: s.abs(), ascending=False)
                        st.dataframe(contrib_df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute contributions: {e}")

        # JSON Report
        st.markdown("### JSON Report for experts:")
        report = {
            "inputs": {
                "age": age_c, "bmi": bmi_c, "FSH": fsh_c, "E2": e2_c,
                "height_cm": float(height_cm), "weight_kg": float(weight_kg),
                "baseline_status": status_choice, "smoking_current": (smokere_bin == 1.0)
            },
            "quantile": {
                "q": 0.5, "label": "median", "months": float(t_median) if np.isfinite(t_median) else None
            },
            "calibrated_risks": {
                str(m): (
                    float(calibrators[str(m)].predict([
                        max(0.0, min(1.0, 1.0 - float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])))
                    ])[0]) if str(m) in calibrators else None
                ) for m in horizons.values()
            },
            "model": {
                "features": features,
                "c_index_best": meta.get("c_index_best"),
                "best_orientation": meta.get("best_orientation"),
                "build_info": meta.get("build_info", {})
            }
        }
        st.download_button(
            "Download",
            data=io.BytesIO(json.dumps(report, indent=2).encode("utf-8")),
            file_name="menopause_prediction.json",
            mime="application/json"
        )

    except Exception as e:
        st.warning(f"Could not compute forecast: {e}")