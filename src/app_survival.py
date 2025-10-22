# src/web_app_survival.py
# Web Streamlit UI: downloads model artifacts from a private GitHub Release via secrets.
# Same UX as local: units (cm/in, kg/lb), Y-axis in %, Months/Years toggle, recommendations, references.

import streamlit as st
import pandas as pd
import numpy as np
import requests, os, io, json, tempfile
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="My Menopause Forecast", layout="wide")

st.markdown("""
<style>
.big-metric { font-size: 2.0rem; font-weight: 700; margin: 0.25rem 0 0.5rem 0; }
.side-note { color: #d14; font-size: 0.95rem; }
.small { font-size: 0.9rem; color: #888; }
h1, h2, h3 { font-weight: 700; }
.block-label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ----- GitHub secrets (set in Streamlit Cloud -> Settings -> Secrets) -----
# GH_OWNER, GH_REPO, GH_TAG (e.g., "model-v1"), GH_TOKEN (repo read access)
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
    """Download artifacts from a private GitHub Release and load them."""
    s = requests.Session()
    s.headers.update({"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
    rel_url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
    r = s.get(rel_url, timeout=30)
    r.raise_for_status()
    release = r.json()

    assets = {a["name"]: a for a in release.get("assets", [])}
    tmpdir = tempfile.mkdtemp(prefix="menopause_artifacts_")
    paths = {}
    for key, name in ASSET_NAMES.items():
        if name not in assets:
            raise FileNotFoundError(f"Asset '{name}' not found in release {tag}")
        asset = assets[name]
        dl = s.get(asset["url"], headers={"Accept": "application/octet-stream"}, timeout=120, stream=True)
        dl.raise_for_status()
        out_path = os.path.join(tmpdir, name)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
        paths[key] = out_path

    model = joblib.load(paths["model"])
    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    calibrators = joblib.load(paths["cal"])
    return model, meta, calibrators

cph, meta, calibrators = download_release_assets(owner, repo, tag, token)

st.title("My Menopause Forecast")

left, right = st.columns([1.1, 1.9])

# ---------- How to use ----------
with left:
    st.header("How to use:")
    st.markdown("""
Enter your health parameters in the boxes according to the instructions down below.
The model will generate your personalized forecast.
You will see a "Predicted median time" and a graph showing how the probability of entering menopause changes over time.
    """)
    st.caption("""
This tool does NOT provide medical advice. Use it for educational purposes only.
It is a statistical estimate, not a guarantee. Please consult your doctor with any personal health questions.
This tool is intended for users who DO NOT have a health condition (other than perimenopause itself) that affects menstrual regularity.
(On mobile, the inputs may appear below; model notes are usually placed in a footnote.)
    """)
    st.markdown("**PROVIDED HEALTH PARAMETERS:**")
    with st.expander("Menopause stage"):
        st.markdown("""
- Pre-menopausal: Had bleeding in the last 3 months and periods "stayed the same".
- Early perimenopause: Had bleeding in the last 3 months, but periods had become "farther apart, closer together, more variable, or more regular".
- Late perimenopause: Had bleeding in the last 12 months, but no bleeding in the last 3 months.
        """)
    with st.expander("Anthropometrics"):
        st.markdown("""
Height between ~140 and 200 cm or 55–79 inches.
Weight between ~40 and 160 kg or 88–353 pounds.

Note: You can enter height in cm or inches and weight in kg or pounds; BMI is auto-calculated.
        """)
    with st.expander("Hormones"):
        st.markdown("""
Please provide values measured on days 2–5 of your cycle if your cycle is regular (premenopause),
or from any day if your cycle is irregular (perimenopause).

FSH (mIU/mL): 0–300 mIU/mL (typical range in the tool).  
Estradiol (E2, pg/mL): 0–400 pg/mL (typical range in the tool).
        """)

# ---------- Helpers ----------
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

def to_cm(value: float, unit: str) -> float:
    return float(value) * 2.54 if unit == "in" else float(value)

def to_kg(value: float, unit: str) -> float:
    return float(value) * 0.45359237 if unit in ("lb", "lbs") else float(value)

def calc_bmi(height_value: float, h_unit: str, weight_value: float, w_unit: str) -> float:
    height_cm = to_cm(height_value, h_unit)
    weight_kg = to_kg(weight_value, w_unit)
    h_m = max(0.5, height_cm / 100.0)
    return float(weight_kg) / (h_m * h_m)

status_label_to_ord = {
    "Pre-menopausal (regular cycles)": 2,
    "Early perimenopausal (cycle irregularity, <3m gap)": 1,
    "Late perimenopausal (>3m gap, <12m)": 0,
}

# ---------- Inputs ----------
with right:
    c1, c2 = st.columns(2)
    with c1:
        status_choice = st.selectbox("Menopause stage", list(status_label_to_ord.keys()), index=1)
        age = st.number_input("Age (years, 35–60)", min_value=35, max_value=60, value=46, step=1, format="%d")
        smoke_choice = st.selectbox("Currently smoking?", ["No", "Yes"], index=0)
    with c2:
        fsh = st.number_input("FSH (mIU/mL)", min_value=0.0, max_value=300.0, value=20.0, step=0.1)
        e2  = st.number_input("Estradiol (pg/mL)", min_value=0.0, max_value=400.0, value=70.0, step=0.1)

    st.markdown("**Anthropometrics**")
    c3, c4, c5 = st.columns([1, 1, 1])
    with c3:
        h_unit = st.selectbox("Height unit", ["cm", "in"], index=0)
        h_min, h_max, h_def = (140.0, 200.0, 164.0) if h_unit == "cm" else (55.0, 79.0, 65.0)
        height_val = st.number_input(f"Height ({h_unit})", min_value=h_min, max_value=h_max, value=h_def, step=0.1)
    with c4:
        w_unit = st.selectbox("Weight unit", ["kg", "lb"], index=0)
        w_min, w_max, w_def = (40.0, 160.0, 62.0) if w_unit == "kg" else (88.0, 353.0, 137.0)
        weight_val = st.number_input(f"Weight ({w_unit})", min_value=w_min, max_value=w_max, value=w_def, step=0.1)
    with c5:
        manual_bmi = st.checkbox("Override BMI", value=False)

    if manual_bmi:
        bmi_raw = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=60.0, value=26.5, step=0.1, help="Manual override")
    else:
        bmi_raw = calc_bmi(height_val, h_unit, weight_val, w_unit)
        st.metric("Calculated BMI (kg/m²)", f"{bmi_raw:.1f}")

    # Show entered values for clarity
    height_cm = to_cm(height_val, h_unit)
    weight_kg = to_kg(weight_val, w_unit)
    st.caption(f"Entered height: {height_cm:.1f} cm ({height_cm/2.54:.1f} in); weight: {weight_kg:.1f} kg ({weight_kg/0.45359237:.1f} lb).")

    # Prepare model inputs (clipped)
    age_c = clip_by_meta("age", float(age))
    bmi_c = clip_by_meta("bmi", float(bmi_raw))
    fsh_c = clip_by_meta("fsh", float(fsh))
    e2_c  = clip_by_meta("e2",  float(e2))
    smokere_bin = 1.0 if smoke_choice == "Yes" else 0.0

    ratio = float(fsh_c / (e2_c + 1e-6))
    r_p99 = meta.get("ratio_p99", None)
    if r_p99 is not None:
        ratio = min(ratio, r_p99)

    features = meta.get("features", ["age", "bmi", "fsh_log", "e2_log", "fsh_e2_ratio", "baseline_status_ord"])
    defaults = meta.get("defaults", {})
    row = {
        "age": age_c,
        "bmi": bmi_c,
        "fsh_log": np.log1p(max(fsh_c, 0.0)),
        "e2_log":  np.log1p(max(e2_c, 0.0)),
        "fsh_e2_ratio": ratio,
        "baseline_status_ord": status_label_to_ord[status_choice],
        "smokere_bin": smokere_bin,
    }
    for f in features:
        if f not in row:
            row[f] = defaults.get(f, 0.0)
    X = pd.DataFrame([row], columns=features)

    # Timeline / horizons
    horizons = {"1-year": 12, "3-year": 36, "5-year": 60}
    timeline_months = np.linspace(0, 360, 361)
    axis_unit = st.radio("Timeline units", ["Years", "Months"], index=0, horizontal=True)

    # --------- Inference and outputs ---------
    try:
        surv = cph.predict_survival_function(X, times=timeline_months)
        s_vals = surv.iloc[:, 0].values
        s_pct  = s_vals * 100.0

        idx = np.where(s_vals <= 0.5)[0]
        t_median_m = float(timeline_months[idx[0]]) if len(idx) > 0 else np.nan

        if np.isfinite(t_median_m):
            st.markdown(f'<div class="big-metric">Predicted median time: {t_median_m:.0f} months (~{t_median_m/12:.1f} years)</div>', unsafe_allow_html=True)
            st.markdown('<div class="side-note">By this time, 50% of women with identical parameters are predicted to have entered menopause.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="big-metric">Predicted median time: > 30 years</div>', unsafe_allow_html=True)

        with st.expander("Detailed Prognosis"):
            cols = st.columns(len(horizons))
            for i, (label, m) in enumerate(horizons.items()):
                s_t = float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])
                p_raw = max(0.0, min(1.0, 1.0 - s_t))
                if str(m) in calibrators:
                    p_cal = float(np.clip(calibrators[str(m)].predict([p_raw])[0], 0.0, 1.0))
                    cols[i].metric(label, f"{p_cal*100:.1f}%", help=f"calibrated (raw: {p_raw*100:.1f}%)")
                else:
                    cols[i].metric(label, f"{p_raw*100:.1f}%", help="raw")

            fig, ax = plt.subplots()
            x_vals = timeline_months/12.0 if axis_unit == "Years" else timeline_months
            ax.plot(x_vals, s_pct, color="#2b6cb0", linewidth=2.0, label="Still not menopausal (%)")
            for yrs, m in [(1,12), (3,36), (5,60)]:
                xv = yrs if axis_unit == "Years" else m
                ax.axvline(xv, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
                s_t = float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])
                p_evt = (1.0 - s_t) * 100.0
                ax.annotate(f"{p_evt:.1f}%", xy=(xv, max(5, s_t*100.0)),
                            xytext=(xv+0.1 if axis_unit=="Years" else xv+4, min(95, s_t*100.0+15)),
                            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6),
                            fontsize=9, color="gray")
            ax.set_xlabel("Years" if axis_unit == "Years" else "Months")
            ax.set_ylabel("% of women who haven’t gone through menopause yet")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            st.pyplot(fig)

        with st.expander("Recommendations"):
            if np.isfinite(t_median_m) and t_median_m < 60:
                st.markdown("""
Prognosis: As women go through the menopausal transition, they commonly experience symptoms such as hot flashes, vaginal dryness, sleep disturbances, and mood changes [1]. This phase is also associated with an increased risk of various long-term health conditions [2, 3]. Because of these risks, timely awareness and management are important.

Your estimated time to menopause is less than 5 years. If you are already experiencing any symptoms, please seek medical advice from your doctor.
                """)

            recs = []
            if smokere_bin >= 0.5:
                recs.append({
                    "title": "Smoking",
                    "text": "Smoking is associated in observational studies with earlier natural menopause. Quitting improves overall health [4, 5]. Consider discussing cessation strategies with a clinician.",
                    "links": [
                        "https://www.cdc.gov/tobacco/index.htm",
                        "https://www.who.int/health-topics/tobacco"
                    ]
                })
            if float(bmi_raw) < 18.5:
                recs.append({
                    "title": "Body weight",
                    "text": "Very low BMI can be associated with hormonal irregularities [7]. Balanced nutrition and adequate energy intake support overall reproductive health.",
                    "links": [
                        "https://www.womenshealth.gov/menopause/menopause-symptoms-and-relief"
                    ]
                })
            elif float(bmi_raw) >= 30.0:
                recs.append({
                    "title": "Lifestyle",
                    "text": "Healthy lifestyle (balanced diet, physical activity, sleep, stress management) supports overall cardiovascular and metabolic health [6].",
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
            st.caption("Educational content only. Not medical advice.")

        with st.expander("References & Sources"):
            st.markdown("""
1. The Menopause Society (NAMS). Menopause Symptoms. https://menopause.org/patient-education/menopause-topics/symptoms  
2. Collaborative Group on Hormonal Factors in Breast Cancer. Body mass index, serum sex hormones, and breast cancer risk in postmenopausal women. Journal of the National Cancer Institute. DOI: https://doi.org/10.1002/ijc.11598  
3. International Agency for Research on Cancer (IARC). Age at Menopause and Cancer Risk. The Lancet Oncology. DOI: https://doi.org/10.1016/S1470-2045(12)70425-4  
4. World Health Organization (WHO). Tobacco. https://www.who.int/health-topics/tobacco  
5. Centers for Disease Control and Prevention (CDC). Health Effects of Smoking and Tobacco Use. https://www.cdc.gov/tobacco/index.htm  
6. World Health Organization (WHO). Healthy Diet. https://www.who.int/news-room/fact-sheets/detail/healthy-diet  
7. Office on Women's Health, U.S. Department of Health & Human Services. Menopause symptoms and relief. https://www.womenshealth.gov/menopause/menopause-symptoms-and-relief
            """)

        with st.expander("Technical Details"):
            feats = meta.get("features", features)
            st.markdown("Cox proportional hazards model trained on SWAN visits (V1–V10).")
            st.markdown(f"Model features: {', '.join(feats)}")
            st.markdown(f"C-index (best): {meta.get('c_index_best', '0.8208172280508007')}")
            bi = meta.get("build_info", {})
            st.markdown(f"Trained at (UTC): {bi.get('trained_at_utc', 'NA')}")
            if st.checkbox("Show factor contribution table (log-hazard)"):
                try:
                    coefs = getattr(cph, "params_", None)
                    if coefs is None:
                        st.info("Model coefficients are not available.")
                    else:
                        coefs = coefs.reindex(feats)
                        xv = X.iloc[0].reindex(feats).astype(float)
                        contrib = coefs.values * xv.values
                        df = pd.DataFrame({"feature": feats, "coef": coefs.values, "value": xv.values, "contribution": contrib})
                        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["contribution"]).sort_values("contribution", key=lambda s: s.abs(), ascending=False)
                        st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute contributions: {e}")

        st.markdown("### JSON Report for experts:")
        report = {
            "inputs": {
                "age": age_c, "bmi": bmi_c, "FSH": fsh_c, "E2": e2_c,
                "height_cm": float(height_cm), "weight_kg": float(weight_kg),
                "height_unit": h_unit, "weight_unit": w_unit,
                "baseline_status": status_choice,
                "smoking_current": (smokere_bin == 1.0)
            },
            "quantile": {"q": 0.5, "label": "median", "months": float(t_median_m) if np.isfinite(t_median_m) else None},
            "calibrated_risks": {
                str(m): (
                    float(calibrators[str(m)].predict([
                        max(0.0, min(1.0, 1.0 - float(cph.predict_survival_function(X, times=[m]).iloc[0, 0])))
                    ])[0]) if str(m) in calibrators else None
                ) for m in horizons.values()
            },
            "model": {
                "features": feats,
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