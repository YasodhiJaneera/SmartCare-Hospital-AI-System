
"""SmartCare – Appointment No-Show Predictor (Task 08 prototype)."""
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import CAT, NUM, build_bundle, predict_patient

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_NAME = "smartcare_ai_dataset_1000.csv"
CSV_CANDIDATES = [os.path.join(HERE, CSV_NAME),
                  os.path.join(HERE, "..", "dataset", "Raw", CSV_NAME),
                  os.path.join("/content", CSV_NAME)]

st.set_page_config(page_title="SmartCare No-Show Predictor", page_icon="🏥")


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    path = os.path.join(HERE, "model_bundle.joblib")
    try:
        return joblib.load(path)
    except Exception:  # missing file or scikit-learn version mismatch -> rebuild
        csv = next(p for p in CSV_CANDIDATES if os.path.exists(p))
        return build_bundle(csv)


def risk_band(p):
    if p >= 0.60:
        return "HIGH", "error", ("Call the patient and send an SMS reminder 48 h before. "
                                 "Offer rescheduling and consider overbooking this slot.")
    if p >= 0.40:
        return "MEDIUM", "warning", "Send an SMS/WhatsApp reminder 24 h before the appointment."
    return "LOW", "success", "Standard automated reminder is sufficient."


def explain(bundle, A):
    """SHAP values for the 'no-show' class, grouped back to the original input variables."""
    import shap
    sv = shap.TreeExplainer(bundle["model"]).shap_values(A)
    sv = sv[1][0] if isinstance(sv, list) else (sv[0, :, 1] if np.ndim(sv) == 3 else sv[0])
    grouped = {}
    for name, v in zip(bundle["feature_names"], sv):
        var = next((c for c in NUM + CAT if name == f"num__{c}" or name.startswith(f"cat__{c}_")), name)
        grouped[var] = grouped.get(var, 0.0) + float(v)
    return pd.Series(grouped).sort_values(key=abs)


bundle = load_bundle()
opt, rng, m = bundle["options"], bundle["ranges"], bundle["metrics"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("About the model")
    st.write("Random Forest (best test F1 in Task 06) trained on the SmartCare dataset.")
    st.metric("Test accuracy", f"{m['Accuracy']:.0%}")
    st.metric("F1 score", f"{m['F1 Score']:.2f}")
    st.metric("ROC-AUC", f"{m['ROC-AUC']:.2f}")
    st.caption("Trained on a synthetic dataset – for academic demonstration only.")

# ---------------- 1. ACCEPT USER INPUT ----------------
st.title("🏥 Appointment No-Show Predictor")
st.write("Enter the patient's appointment details to estimate the chance they will miss it.")

with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age (years)", rng["age"][0], 100, 35)
        gender = st.selectbox("Gender", opt["gender"])
        blood_group = st.selectbox("Blood group", opt["blood_group"])
        department = st.selectbox("Department", opt["department"])
        diagnosis = st.selectbox("Diagnosis", opt["diagnosis"])
    with c2:
        waiting_days = st.slider("Waiting days (booking → appointment)", 0, 60, 10)
        prev_app = st.slider("Previous appointments", 0, 20, 3)
        missed = st.slider("Previously missed appointments", 0, 20, 0)
        prev_adm = st.slider("Previous admissions", 0, 10, 0)
    submitted = st.form_submit_button("Predict no-show risk", type="primary")

if submitted:
    if missed > prev_app:
        st.error("Missed appointments cannot exceed previous appointments.")
        st.stop()

    # ---------------- 2. PROCESS PATIENT INFORMATION ----------------
    patient = {"age": age, "gender": gender, "blood_group": blood_group,
               "department": department, "diagnosis": diagnosis,
               "waiting_days": waiting_days, "previous_appointments": prev_app,
               "missed_previous_appointments": missed, "previous_admissions": prev_adm}

    # ---------------- 3. GENERATE PREDICTION ----------------
    prob, A = predict_patient(bundle, patient)
    level, kind, action = risk_band(prob)

    # ---------------- 4. DISPLAY RESULTS ----------------
    st.subheader("Prediction result")
    st.metric("No-show probability", f"{prob:.1%}",
              delta=f"{(prob - bundle['base_rate']) * 100:+.1f} pts vs. average patient",
              delta_color="inverse")
    st.progress(min(max(prob, 0.0), 1.0))
    getattr(st, kind)(f"**{level} RISK** – {action}")

    st.subheader("Why this prediction?")
    try:
        contrib = explain(bundle, A)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(contrib.index, contrib.values,
                color=["#d62728" if v > 0 else "#2ca02c" for v in contrib.values])
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("SHAP value  (red = raises no-show risk, green = lowers it)")
        fig.tight_layout()
        st.pyplot(fig)
    except Exception:
        imp = pd.Series(bundle["model"].feature_importances_, index=bundle["feature_names"])
        st.bar_chart(imp.sort_values().tail(8))
        st.caption("Showing global feature importance (SHAP unavailable).")

    with st.expander("Input summary"):
        st.table(pd.DataFrame(patient, index=["value"]).T.astype(str))

st.caption("⚠️ Decision-support prototype only. Predictions are probabilities, not certainties.")
