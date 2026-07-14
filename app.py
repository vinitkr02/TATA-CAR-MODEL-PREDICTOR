import streamlit as st
import numpy as np
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🚗 Car Model Predictor",
    page_icon="🚗",
    layout="centered"
)

# ----------------------------
# Custom CSS for Better UI
# ----------------------------
st.markdown("""
    <style>
    /* Main Background & Typography styling */
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1E293B;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
    }
    
    /* Elegant card styling for the results */
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border-left: 5px solid #2563EB;
        margin-top: 20px;
    }
    .result-header {
        color: #1E293B;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .result-value {
        color: #2563EB;
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True) # <-- Fixed keyword here

# ----------------------------
# Load Files
# ----------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load("Logistic_Regression_Batch_2.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, label_encoder

try:
    model, scaler, label_encoder = load_ml_assets()
except Exception as e:
    st.error("Error loading model files. Please verify that the file paths are correct.")
    st.stop()

# ----------------------------
# Header Section
# ----------------------------
st.title("🚗 Car Model Predictor")
st.markdown("Fine-tune the car specifications below using the sliders to instantly predict the exact model using Machine Learning.")
st.divider()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🛠️ App Overview")
st.sidebar.info("""
This dashboard leverages a **Logistic Regression** pipeline to evaluate physical and financial specs, mapping them back to the most probable vehicle model.

**Features Analyzed:**
* Body Type
* Engine Capacity
* Power (BHP)
* Fuel Mileage
* Ex-Showroom Valuation
""")

# ----------------------------
# Input Features Form
# ----------------------------
body_types = ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Pickup", "Van", "Wagon"]
body_mapping = {b: i for i, b in enumerate(body_types)}

col1, col2 = st.columns(2)

with col1:
    body_type = st.selectbox("📋 Body Type", body_types)
    
    engine = st.slider(
        "🔧 Engine Capacity (cc)",
        min_value=500,
        max_value=8000,
        value=1500,
        step=50,
        help="Select displacement capacity in cubic centimeters."
    )
    
    power = st.slider(
        "⚡ Power (BHP)",
        min_value=20,
        max_value=1200,
        value=100,
        step=5,
        help="Brake Horsepower maximum output limit."
    )

with col2:
    mileage = st.slider(
        "⛽ Mileage (km/l)",
        min_value=5.0,
        max_value=40.0,
        value=18.5,
        step=0.5,
        help="Expected fuel economy range."
    )
    
    ex_showroom_price = st.slider(
        "💰 Ex-Showroom Price (₹ Lakh)",
        min_value=1.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Base ex-showroom valuation estimate."
    )

st.markdown("<br>", unsafe_allow_html=True) # <-- Fixed keyword here

# ----------------------------
# Prediction Processing
# ----------------------------
if st.button("🔮 Predict Car Model", use_container_width=True):
    
    features = np.array([[
        body_mapping[body_type],
        engine,
        power,
        mileage,
        ex_showroom_price
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    predicted_model = label_encoder.inverse_transform(prediction)
    
    probability = model.predict_proba(features_scaled)
    confidence = np.max(probability) * 100

    # Custom HTML Styled Output block
    st.markdown(f"""
        <div class="result-card">
            <div class="result-header">🚘 Predicted Match</div>
            <div class="result-value">{predicted_model[0]}</div>
            <hr style="margin: 15px 0; border: 0; border-top: 1px solid #E2E8F0;">
            <div style="display: flex; justify-content: space-between;">
                <div><strong>Model Type:</strong> {body_type}</div>
                <div><strong>Confidence Index:</strong> {confidence:.2f}%</div>
            </div>
        </div>
    """, unsafe_allow_html=True) # <-- Fixed keyword here

st.divider()
st.caption("Built with ❤️ using Python, Scikit-learn & Streamlit")