import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_files():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_files()

st.markdown("""
<style>

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.main-title{
    font-size:42px;
    font-weight:700;
    color:#0E7490;
    text-align:center;
}

.subtitle{
    font-size:18px;
    color:#666;
    text-align:center;
    margin-bottom:25px;
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 2px 15px rgba(0,0,0,.08);
    border:1px solid #ECECEC;
}


.high-risk{
    background:#FFF1F2;
    border-left:8px solid #DC2626;
    padding:20px;
    border-radius:12px;
    color: black;
}

.low-risk{
    background:#ECFDF5;
    border-left:8px solid #16A34A;
    padding:20px;
    border-radius:12px;
    color: black;
}


.metric{
    background:#F8FAFC;
    padding:15px;
    border-radius:10px;
    text-align:center;
    border:1px solid #E5E7EB;
}


.footer{
    text-align:center;
    color:gray;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/heart-with-pulse.png",
        width=90,
    )

    st.title("Heart Disease Prediction")

    st.caption(
        "Machine Learning based prediction system using the "
        "K-Nearest Neighbors (KNN) algorithm."
    )

    st.divider()

    st.subheader("📊 Model Performance")

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            label="Accuracy",
            value="88.59%"
        )

    with c2:
        st.metric(
            label="F1 Score",
            value="89.86%"
        )

    st.divider()

    st.subheader("⚙️ Tech Stack")

    st.markdown("""
                - Python
                - Pandas
                - NumPy
                - Scikit-learn
                - Streamlit
                """)

    st.divider()
    st.subheader("🔗 Project")

    st.link_button(
        "🌐 Live Demo",
        "https://heart-disease-predict-404.streamlit.app/"
    )

    st.link_button(
        "💻 GitHub Repository",
        "https://github.com/Sathvara-Amitkumar/Heart-Disease-Prediction"
    )

    st.divider()
    st.info(
        """
            This prediction is generated using a trained
            Machine Learning model and is intended for
            educational purposes only.

            It is **not** a medical diagnosis.
        """
    )

    st.divider()
    st.caption(
        "Developed by\n\n**Amitkumar Sathvara**"
    )


# =====================================
# HERO SECTION
# =====================================

st.markdown(
    """
    <div class="main-title">
        ❤️ Heart Disease Prediction System
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")



# Part - 4

# =====================================
# SESSION STATE
# =====================================

DEFAULT_VALUES = {
    "age": 30,
    "sex": "Male",
    "resting_bp": 120,
    "cholesterol": 200,
    "fasting_bs": 0,
    "max_hr": 150,
    "oldpeak": 1.0,
    "chest_pain": "ATA",
    "resting_ecg": "Normal",
    "exercise_angina": "N",
    "st_slope": "Up"
}

if "initialized" not in st.session_state:
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value
    st.session_state.initialized = True


def load_healthy_example():

    st.session_state.age = 28
    st.session_state.sex = "Female"
    st.session_state.resting_bp = 110
    st.session_state.cholesterol = 170
    st.session_state.fasting_bs = 0
    st.session_state.max_hr = 180
    st.session_state.oldpeak = 0.2
    st.session_state.chest_pain = "ATA"
    st.session_state.resting_ecg = "Normal"
    st.session_state.exercise_angina = "N"
    st.session_state.st_slope = "Up"

def load_high_risk():

    st.session_state.age = 65
    st.session_state.sex = "Male"
    st.session_state.resting_bp = 170
    st.session_state.cholesterol = 320
    st.session_state.fasting_bs = 1
    st.session_state.max_hr = 90
    st.session_state.oldpeak = 4.5
    st.session_state.chest_pain = "ASY"
    st.session_state.resting_ecg = "ST"
    st.session_state.exercise_angina = "Y"
    st.session_state.st_slope = "Flat"


def reset_form():

    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value


st.write("")