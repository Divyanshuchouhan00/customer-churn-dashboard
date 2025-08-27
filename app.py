# app.py — Customer Churn Dashboard with AI Agent (enhanced)

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional animation
from streamlit_lottie import st_lottie
import requests

# Project modules
from src.preprocessing import load_data, encode_features
from src.model import train_model, predict_churn
from src.visualizations import (
    plot_monthly_charges,
    plot_tenure_churn,
    plot_correlation_heatmap,
    plot_confusion_matrix
)

# Explainability
import shap

# --------------- Page config & theme tweaks ---------------
st.set_page_config(page_title="Customer Churn • AI Agent", layout="wide")
st.markdown("""
<style>
.stApp { background:#0e1117; color:#fff; }
.block-container { padding-top: 1.2rem; }
.metric { text-align:center; }
</style>
""", unsafe_allow_html=True)

# --------------- Lottie (optional) ---------------
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

lottie = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_tQ3n6m.json")

# --------------- Title ---------------
c1, c2 = st.columns([4,1])
with c1:
    st.title("📉 Customer Churn Dashboard — AI Agent")
with c2:
    if lottie:
        st_lottie(lottie, height=110, key="churn-anim")

# --------------- Load + prep data / model ---------------
@st.cache_data(show_spinner=False)
def get_raw():
    return load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_data(show_spinner=False)
def get_encoded(df_raw: pd.DataFrame):
    return encode_features(df_raw)

@st.cache_resource(show_spinner=False)
def get_model(df_encoded: pd.DataFrame):
    return train_model(df_encoded)

df_raw = get_raw()
df_encoded = get_encoded(df_raw)
model = get_model(df_encoded)

# Feature matrix & labels
X = df_encoded.drop(columns=["Churn"])
y_true = df_encoded["Churn"].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

# --------------- Sidebar: filters + agent input ---------------
st.sidebar.header("🔍 Filters")
gender_f = st.sidebar.selectbox("Gender", ["All"] + sorted(df_raw["gender"].unique().tolist()))
senior_f = st.sidebar.selectbox("Senior Citizen", ["All"] + sorted(df_raw["SeniorCitizen"].unique().tolist()))
contract_f = st.sidebar.selectbox("Contract Type", ["All"] + sorted(df_raw["Contract"].unique().tolist()))
tenure_f = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))
monthly_f = st.sidebar.slider("Monthly Charges", float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max()), (0.0, 120.0))

st.sidebar.header("🤖 AI Agent — Predict Single Customer")
threshold = st.sidebar.slider("High-Risk Threshold", 0.0, 1.0, 0.70, 0.01)
gender_in = st.sidebar.selectbox("Input Gender", ["Male", "Female"])
senior_in = st.sidebar.selectbox("Input Senior Citizen", [0, 1])
contract_in = st.sidebar.selectbox("Input Contract Type", ["Month-to-month", "One year", "Two year"])
tenure_in = st.sidebar.slider("Input Tenure (months)", 0, 72, 12)
monthly_in = st.sidebar.slider("Input Monthly Charges", 0.0, 120.0, 50.0)

# Build a model-ready row from raw-like inputs
def build_feature_row(X_cols, tenure, monthly, senior, gender, contract):
    row = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "SeniorCitizen": senior,
        # one-hots we know exist from previous work
        "gender_Male": 1 if gender == "Male" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
    }
    df_row = pd.DataFrame([row])
    # Match all model columns, fill missing with 0
    return df_row.reindex(columns=X_cols, fill_value=0)

input_df = build_feature_row(X.columns, tenure_in, monthly_in, senior_in, gender_in, contract_in)

# --------------- Helper: probability & prediction ---------------
def predict_proba(model, X_row: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_row)[0, 1])
    # Fallback for decision_function-only models
    if hasattr(model, "decision_function"):
        from sklearn.preprocessing import MinMaxScaler
        score = model.decision_function(X_row).reshape(-1, 1)
        return float(MinMaxScaler().fit_transform(score)[0, 0])
    # Last resort
    return float(model.predict(X_row)[0])

# --------------- Helper: recommendations ---------------
def generate_recommendations(prob: float, raw_inputs: dict) -> list:
    recs = []
    # Global risk-based
    if prob >= 0.8:
        recs.append("Immediate retention call within 24h.")
        recs.append("Offer 15–20% limited-time discount.")
        recs.append("Escalate to customer success manager.")
    elif prob >= 0.7:
        recs.append("Offer 10% discount or loyalty credits.")
        recs.append("Bundle add-on services to increase value.")
    elif prob >= 0.5:
        recs.append("Send personalized email with upgrade incentives.")
        recs.append("Provide proactive support / training tips.")

    # Feature-aware nudges
    if raw_inputs.get("contract") == "Month-to-month":
        recs.append("💡 Suggest moving to One/Two-year contract for stability.")
    if raw_inputs.get("monthly") and raw_inputs["monthly"] > 80:
        recs.append("💡 Consider bill optimization or discount on add-ons.")
    if raw_inputs.get("tenure") and raw_inputs["tenure"] < 6:
        recs.append("💡 Onboarding outreach: quick-start guide + welcome call.")
    if raw_inputs.get("senior") == 1:
        recs.append("💡 Offer simplified plan options and priority support.")
    if raw_inputs.get("gender") == "Female":
        # purely demonstrative; in real apps avoid sensitive-targeted recs unless justified & compliant
        recs.append("💡 Ensure comms highlight reliability & safety features.")
    # Dedup while preserving order
    seen = set()
    final = []
    for r in recs:
        if r not in seen:
            final.append(r); seen.add(r)
    return final

# --------------- Helper: SHAP explanation ---------------
@st.cache_resource(show_spinner=False)
def get_explainer(_model, X_background: pd.DataFrame):
    # SHAP picks the right algorithm (Tree/Linear/Kernel) for many sklearn models
    return shap.Explainer(_model, X_background)

def compute_shap_topN(explainer, X_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    sv = explainer(X_row)
    # For binary classifiers, values shape: (1, n_features)
    shap_vals = sv.values[0]
    base_val = float(np.atleast_1d(sv.base_values)[0])
    contrib = pd.DataFrame({
        "feature": X_row.columns,
        "value": X_row.iloc[0].values,
        "shap_value": shap_vals
    })
    contrib["abs_val"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_val", ascending=False).head(top_n)
    contrib["direction"] = np.where(contrib["shap_value"] >= 0, "↑ increases churn", "↓ decreases churn")
    return contrib, base_val

# --------------- Apply filters to raw ---------------
df_f = df_raw.copy()
if gender_f != "All":
    df_f = df_f[df_f["gender"] == gender_f]
if senior_f != "All":
    df_f = df_f[df_f["SeniorCitizen"] == senior_f]
if contract_f != "All":
    df_f = df_f[df_f["Contract"] == contract_f]
df_f = df_f[
    (df_f["tenure"].between(tenure_f[0], tenure_f[1])) &
    (df_f["MonthlyCharges"].between(monthly_f[0], monthly_f[1]))
]

# --------------- Top KPIs ---------------
st.markdown("## 🌟 Top Insights")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("👥 Total Customers", f"{len(df_raw):,}")
with k2:
    churned = (df_raw["Churn"] == "Yes").sum()
    st.metric("⚠️ Churned Customers", f"{churned:,}")
with k3:
    churn_rate = (df_raw["Churn"].eq("Yes").mean() * 100)
    st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
with k4:
    st.metric("💳 Avg Monthly Charges", f"${df_raw['MonthlyCharges'].mean():.2f}")

# --------------- Tabs ---------------
tabs = st.tabs([
    "📈 Visual Insights",
    "🔮 AI Explainability",
    "⚡ What-if Simulator",
    "🧩 Risk Segmentation",
    "🧾 Predictions & Download",
    "💬 Ask the Agent"
])

# ========== Tab 1: Visual Insights ==========
with tabs[0]:
    st.subheader("Visuals (filtered)")
    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(plot_monthly_charges(df_f), use_container_width=True)
    with cB:
        st.plotly_chart(plot_tenure_churn(df_f), use_container_width=True)

    st.plotly_chart(plot_correlation_heatmap(df_encoded), use_container_width=True)

    # Confusion matrix on full training set predictions
    y_pred = pd.Series(model.predict(X)).fillna(0).astype(int)
    st.plotly_chart(plot_confusion_matrix(y_true, y_pred), use_container_width=True)

# ========== Tab 2: AI Explainability (SHAP) ==========
with tabs[1]:
    st.subheader("Why did the model predict this? (SHAP)")

    # Show current single-customer inputs + prediction
    in_cols = st.columns(5)
    in_cols[0].metric("Gender", gender_in)
    in_cols[1].metric("Senior", str(senior_in))
    in_cols[2].metric("Contract", contract_in)
    in_cols[3].metric("Tenure", str(tenure_in))
    in_cols[4].metric("Monthly", f"${monthly_in:.2f}")

    prob_single = predict_proba(model, input_df)
    st.metric("Predicted Churn Probability", f"{prob_single:.2f}")
    st.progress(min(1.0, max(0.0, prob_single)))

    with st.spinner("Computing SHAP…"):
        explainer = get_explainer(model, X.sample(min(500, len(X)), random_state=42))  # speed up
        sv = explainer(X.sample(1, random_state=7))
        shap_df = pd.DataFrame({
        "feature": X.columns,
        "shap_value": sv.values[0],
        })
        shap_df["abs_val"] = shap_df["shap_value"].abs()  # absolute impact
        shap_df["direction"] = shap_df["shap_value"].apply(lambda x: "Positive" if x > 0 else "Negative")

        top_features = shap_df.sort_values("abs_val", ascending=False).head(5)
        st.write("Likely top drivers (data & model dependent):")
        st.write(", ".join(top_features["feature"].tolist()))

    # Plot as bar (absolute impact)
    fig_bar = px.bar(
        shap_df.sort_values("abs_val",ascending=True).tail(10),
        x="abs_val", y="feature",
        orientation="h",
        title="Top feature contributions (|SHAP|)",
        hover_data={"shap_value": True, "direction": True, "abs_val": False}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.caption("Positive SHAP → pushes probability ↑ (more churn). Negative SHAP → pushes probability ↓.")

# ========== Tab 3: What-if Simulator ==========
with tabs[2]:
    st.subheader("What-if: simulate plan or price changes")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    contract_sim = sim_c1.selectbox("Sim Contract", ["Same", "Month-to-month", "One year", "Two year"])
    monthly_sim = sim_c2.slider("Sim Monthly Charges", 0.0, 120.0, float(monthly_in))
    tenure_sim = sim_c3.slider("Sim Tenure (months)", 0, 72, int(tenure_in))

    # Build simulated row
    sim_contract = contract_in if contract_sim == "Same" else contract_sim
    sim_row = build_feature_row(
        X.columns,
        tenure=tenure_sim,
        monthly=monthly_sim,
        senior=senior_in,
        gender=gender_in,
        contract=sim_contract
    )
    prob_sim = predict_proba(model, sim_row)

    d1, d2, d3 = st.columns(3)
    d1.metric("Current Prob", f"{prob_single:.2f}")
    d2.metric("Simulated Prob", f"{prob_sim:.2f}")
    delta = prob_sim - prob_single
    d3.metric("Δ Change", f"{delta:+.2f}")

    # Smart recommendations for simulated scenario
    st.markdown("**AI Agent Recommendations (for simulated scenario):**")
    for rec in generate_recommendations(prob_sim, {
        "contract": sim_contract, "monthly": monthly_sim, "tenure": tenure_sim,
        "senior": senior_in, "gender": gender_in
    }):
        st.write("• " + rec)

# ========== Tab 4: Risk Segmentation ==========
with tabs[3]:
    st.subheader("Segment customers by churn risk")
    # Predict for full dataset
    if hasattr(model, "predict_proba"):
        probs_all = model.predict_proba(X)[:, 1]
    else:
        # fallback
        probs_all = pd.Series(model.predict(X)).astype(float).values

    seg = pd.DataFrame({
        "prob": probs_all
    })
    def bucket(p):
        if p >= 0.7: return "High (≥0.70)"
        if p >= 0.3: return "Medium (0.30–0.69)"
        return "Low (<0.30)"
    seg["bucket"] = seg["prob"].apply(bucket)

    c1, c2 = st.columns(2)
    with c1:
        pie = px.pie(seg, names="bucket", title="Risk distribution", hole=0.35)
        st.plotly_chart(pie, use_container_width=True)
    with c2:
        hist = px.histogram(seg, x="prob", nbins=25, title="Churn probability histogram")
        st.plotly_chart(hist, use_container_width=True)

# ========== Tab 5: Predictions & Download ==========
with tabs[4]:
    st.subheader("Batch predictions & export")
    # Join predictions back to raw for context
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = pd.Series(model.predict(X)).astype(float).values
    preds = (probs >= threshold).astype(int)

    out = df_raw.copy()
    out["churn_probability"] = probs
    out["predicted_churn"] = np.where(preds == 1, "Yes", "No")

    # Apply same filters as top
    m = pd.Series(True, index=out.index)
    if gender_f != "All": m &= out["gender"].eq(gender_f)
    if senior_f != "All": m &= out["SeniorCitizen"].eq(senior_f)
    if contract_f != "All": m &= out["Contract"].eq(contract_f)
    m &= out["tenure"].between(tenure_f[0], tenure_f[1])
    m &= out["MonthlyCharges"].between(monthly_f[0], monthly_f[1])

    out_f = out[m].sort_values("churn_probability", ascending=False)

    st.dataframe(out_f.head(500), use_container_width=True)
    csv = out_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered predictions (CSV)",
        data=csv, file_name="churn_predictions_filtered.csv", mime="text/csv"
    )

# ========== Tab 6: Ask the Agent ==========
with tabs[5]:
    st.subheader("Ask the Agent (rule-based mini assistant)")
    q = st.text_input("Ask a question (e.g., 'why is churn high for month-to-month?')")
    if q:
        ql = q.lower()
        answered = False

        if "month-to-month" in ql or "month to month" in ql:
            rate = df_raw[df_raw["Contract"].eq("Month-to-month")]["Churn"].eq("Yes").mean() * 100
            st.write(f"Customers on **Month-to-month** have a churn rate of **{rate:.2f}%**. "
                     f"Longer contracts typically reduce churn by increasing commitment.")
            answered = True

        if ("senior" in ql) or ("older" in ql):
            rate = df_raw[df_raw["SeniorCitizen"].eq(1)]["Churn"].eq("Yes").mean() * 100
            st.write(f"**Senior Citizens** churn at **{rate:.2f}%** in this dataset. "
                     f"Consider simplified plans and priority support.")
            answered = True

        if "monthly charge" in ql or "price" in ql or "billing" in ql:
            high = df_raw[df_raw["MonthlyCharges"] > df_raw["MonthlyCharges"].median()]
            rate = high["Churn"].eq("Yes").mean() * 100
            st.write(f"Above-median **Monthly Charges** group churn rate is **{rate:.2f}%**. "
                     f"Price optimization or discounts may help.")
            answered = True

        if "top" in ql and ("feature" in ql or "reason" in ql or "driver" in ql):
            # quick SHAP over a small sample
            with st.spinner("Analyzing top churn drivers…"):
                explainer = get_explainer(model, X.sample(min(500, len(X)), random_state=42))
                sv = explainer(X.sample(1, random_state=7))
                vals = pd.Series(sv.values[0], index=X.columns).abs().sort_values(ascending=False).head(5)
            st.write("Likely top drivers (data & model dependent):")
            st.write(", ".join(vals.index.tolist()))
            answered = True

        if not answered:
            st.write("I didn’t recognize that yet. Try:\n"
                     "- 'why is churn high for month-to-month?'\n"
                     "- 'senior citizen churn rate'\n"
                     "- 'impact of monthly charges'\n"
                     "- 'top churn drivers'")

# --------------- Sidebar: Single prediction + recs ---------------
if st.sidebar.button("🔮 Predict (Single)"):
    prob = predict_proba(model, input_df)
    pred = int(prob >= threshold)
    st.sidebar.metric("Churn Probability", f"{prob:.2f}")
    st.sidebar.metric("Prediction", "Yes ✅" if pred else "No ❌")

    st.sidebar.markdown("**Agent Recommendations:**")
    for rec in generate_recommendations(prob, {
        "contract": contract_in, "monthly": monthly_in, "tenure": tenure_in,
        "senior": senior_in, "gender": gender_in
    }):
        st.sidebar.write("• " + rec)

# --------------- Footer ---------------
st.markdown("---")
st.caption("Built with ❤️ Streamlit • Plotly • scikit-learn • SHAP")

