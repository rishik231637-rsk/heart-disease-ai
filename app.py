import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Advanced Page Config
st.set_page_config(page_title="HeartGuard AI", page_icon="🫀", layout="wide")

# Custom CSS to make it look modern
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🫀 HeartGuard: Clinical Prediction Dashboard")
st.write("---")

# 2. Optimized Brain (Architecture)
@st.cache_resource
def get_trained_model():
    df = pd.read_csv("heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    # We use more 'trees' (n_estimators) for better accuracy
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, model_columns = get_trained_model()

# 3. Sidebar for Technical Info
with st.sidebar:
    st.header("System Status")
    st.success("Model: Random Forest")
    st.info("Dataset: UCI Heart Disease")
    st.write("---")
    st.header("Instructions")
    st.write("Enter patient vitals and click 'Generate Report'.")

# 4. Two-Column Input Layout (Better UX)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Metrics")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 1, 100, 45)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
        trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    with c2:
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 210)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.radio("Exercise Induced Pain?", ["No", "Yes"])
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

    # Convert readable text to numbers for the AI
    sex_val = 1 if sex == "Male" else 0
    exang_val = 1 if exang == "Yes" else 0
    
    # Matching the exact order the model expects
    input_data = [age, sex_val, cp, trestbps, chol, 0, 0, thalach, exang_val, oldpeak, 1, 0, 2]

with col2:
    st.subheader("AI Analysis")
    if st.button("Generate Report"):
        # Get Probability
        prob = model.predict_proba([input_data])[0]
        risk_score = prob[1] * 100  # Percentage of risk
        
        # Display Result with a Progress Bar
        st.metric(label="Risk Probability", value=f"{risk_score:.1f}%")
        st.progress(risk_score / 100)
        
        if risk_score > 50:
            st.error("### HIGH RISK DETECTED")
            st.warning("Immediate clinical consultation recommended.")
        else:
            st.success("### LOW RISK")
            st.info("Patient appears to be in a healthy range.")
            
        # Add a chart showing what the AI sees
        chart_data = pd.DataFrame({"Result": ["Healthy", "At Risk"], "Probability": prob})
        st.bar_chart(chart_data.set_index("Result"))