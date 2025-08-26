import streamlit as st
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# Custom imports
from src.preprocessing import load_data, encode_features
from src.model import train_model, predict_churn
from src.visualizations import (
    plot_monthly_charges, 
    plot_tenure_churn, 
    plot_correlation_heatmap,
    plot_confusion_matrix
)

# ---------------------- Page Styling ----------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Lottie Animation ----------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_churn = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_tQ3n6m.json")

# ---------------------- Title ----------------------
col1, col2 = st.columns([4,1])
with col1:
    st.title("📉 Customer Churn Analysis Dashboard")
with col2:
    if lottie_churn:
        st_lottie(lottie_churn, height=120, key="churn")

# ---------------------- Sidebar Filters ----------------------
st.sidebar.header("🔍 Customer Filters")
gender = st.sidebar.selectbox("Gender", ["All","Male","Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["All",0,1])
contract = st.sidebar.selectbox("Contract Type", ["All","Month-to-month","One year","Two year"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, (0,72))
monthly = st.sidebar.slider("Monthly Charges", 0.0, 120.0, (0.0,120.0))
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.5)

# ---------------------- Load and preprocess data ----------------------
df_raw = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_encoded = encode_features(df_raw)

# ---------------------- Train model ----------------------
model = train_model(df_encoded)
X = df_encoded.drop('Churn', axis=1)

# ---------------------- Sidebar Prediction ----------------------
st.sidebar.header("⚡ Predict Churn for Custom Input")
gender_input = st.sidebar.selectbox("Input Gender", ["Male", "Female"])
senior_input = st.sidebar.selectbox("Input Senior Citizen", [0,1])
contract_input = st.sidebar.selectbox("Input Contract Type", ["Month-to-month","One year","Two year"])
tenure_input = st.sidebar.slider("Input Tenure (months)", 0, 72, 12)
monthly_input = st.sidebar.slider("Input Monthly Charges", 0.0, 120.0, 50.0)

input_dict = {
    'tenure': tenure_input,
    'MonthlyCharges': monthly_input,
    'SeniorCitizen': senior_input,
    'gender_Male': 1 if gender_input == 'Male' else 0,
    'Contract_One year': 1 if contract_input == 'One year' else 0,
    'Contract_Two year': 1 if contract_input == 'Two year' else 0
}
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

if st.sidebar.button("Predict Churn"):
    pred, prob = predict_churn(model, input_df, threshold)
    st.sidebar.metric(label="Churn Prediction", value="Yes ✅" if pred else "No ❌")
    st.sidebar.metric(label="Churn Probability", value=f"{prob:.2f}")

# ---------------------- Top Insights Cards ----------------------
st.markdown("## 🌟 Top Insights")
cards_col1, cards_col2, cards_col3, cards_col4 = st.columns(4)

with cards_col1:
    st.image("assets/Screenshot1.png", caption="Customer Distribution", use_container_width=True)
    st.markdown(f"**Total Customers:** {df_raw.shape[0]:,}")
with cards_col2:
    st.image("assets/Screenshot2.png", caption="Churned Customers", use_container_width=True)
    st.markdown(f"**Churned:** {df_raw[df_raw['Churn']=='Yes'].shape[0]:,}")
with cards_col3:
    st.image("assets/Screenshot3.png", caption="Churn Rate", use_container_width=True)
    st.markdown(f"**Rate:** {df_raw['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")
with cards_col4:
    st.image("assets/Screenshot4.png", caption="Average Monthly Charges", use_container_width=True)
    st.markdown(f"**Avg Charges:** ${df_raw['MonthlyCharges'].mean():.2f}")

# ---------------------- KPIs ----------------------
st.markdown("## 📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("👥 Total Customers", df_raw.shape[0])
col2.metric("⚠️ Churned Customers", df_raw[df_raw['Churn']=='Yes'].shape[0])
col3.metric("📉 Churn Rate", f"{df_raw['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

# ---------------------- Visual Insights Tabs ----------------------
st.markdown("## 📈 Visual Insights")

df_filtered = df_raw.copy()
if gender != "All":
    df_filtered = df_filtered[df_filtered['gender']==gender]
if senior != "All":
    df_filtered = df_filtered[df_filtered['SeniorCitizen']==senior]
if contract != "All":
    df_filtered = df_filtered[df_filtered['Contract']==contract]
df_filtered = df_filtered[
    (df_filtered['tenure'] >= tenure[0]) & (df_filtered['tenure'] <= tenure[1]) &
    (df_filtered['MonthlyCharges'] >= monthly[0]) & (df_filtered['MonthlyCharges'] <= monthly[1])
]

# Prepare confusion matrix values
y_true = df_encoded['Churn'].map({'No':0,'Yes':1}).fillna(0).astype(int)
y_pred = pd.Series(model.predict(X)).fillna(0).astype(int)

tab1, tab2, tab3, tab4 = st.tabs([
    "Monthly Charges", 
    "Tenure vs Churn", 
    "Correlation Heatmap",
    "Confusion Matrix"
])

with tab1:
    st.plotly_chart(plot_monthly_charges(df_filtered), use_container_width=True)
with tab2:
    st.plotly_chart(plot_tenure_churn(df_filtered), use_container_width=True)
with tab3:
    st.plotly_chart(plot_correlation_heatmap(df_encoded), use_container_width=True)
with tab4:
    st.plotly_chart(plot_confusion_matrix(y_true, y_pred), use_container_width=True)

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("✨ Built with ❤️ using Streamlit")

