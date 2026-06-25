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



btn1, btn2, btn3 = st.columns(3)

with btn1:
    st.button(
        "🟢 Healthy Example",
        use_container_width=True,
        on_click=load_healthy_example
    )

with btn2:
    st.button(
        "🔴 High Risk Example",
        use_container_width=True,
        on_click=load_high_risk
    )

with btn3:
    st.button(
        "🔄 Reset",
        use_container_width=True,
        on_click=reset_form
    )

st.write("")


# Form

left, right = st.columns(2)

with left:

    age = st.slider(
        "Age",
        18,
        100,
        key="age"
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        50,
        250,
        key="resting_bp"
    )

    cholesterol = st.number_input(
        "Cholesterol",
        0,
        700,
        key="cholesterol"
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60,
        220,
        key="max_hr"
    )

    oldpeak = st.slider(
        "Old Peak",
        0.0,
        10.0,
        step=0.1,
        key="oldpeak"
    )

with right:

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"],
        key="sex"
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar >120",
        [0,1],
        key="fasting_bs"
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA","NAP","TA","ASY"],
        key="chest_pain"
    )

    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal","ST","LVH"],
        key="resting_ecg"
    )

    exercise_angina = st.selectbox(
        "Exercise Angina",
        ["N","Y"],
        key="exercise_angina"
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up","Flat","Down"],
        key="st_slope"
    )



# =====================================
# PREPARE INPUT
# =====================================

input_data = {
    "Age": age,
    "RestingBP": resting_bp,
    "Cholesterol": cholesterol,
    "FastingBS": fasting_bs,
    "MaxHR": max_hr,
    "Oldpeak": oldpeak,

    "Sex_M": 1 if sex == "Male" else 0,

    "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
    "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
    "ChestPainType_TA": 1 if chest_pain == "TA" else 0,

    "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
    "RestingECG_ST": 1 if resting_ecg == "ST" else 0,

    "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,

    "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
    "ST_Slope_Up": 1 if st_slope == "Up" else 0,
}

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(columns=columns, fill_value=0)

scaled_input = scaler.transform(input_df)


st.write("")
st.write("")

predict = st.button(
    "❤️ Predict Heart Disease",
    use_container_width=True,
    type="primary"
)



if predict:

    with st.spinner("Analyzing patient data..."):
        prediction = model.predict(scaled_input)[0]
        probability = None

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input)[0]

        st.write("")
        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.markdown(
                """
                <div class="low-risk">
                <h2>🟢 Low Risk</h2>
                <h4>
                    The model predicts a lower likelihood of heart disease.
                </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )   

        else:
            st.markdown(
                """
                <div class="high-risk">
                <h2>🔴 High Risk</h2>
                <h4>
                The model predicts a higher likelihood of heart disease.
                Please consult a qualified healthcare professional.
                </h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if probability is not None:

            st.write("")
            st.subheader("Prediction Confidence")

            healthy = probability[0]
            disease = probability[1]

            st.write("🟢 Healthy")
            st.progress(float(healthy))
            st.write(f"{healthy*100:.2f}%")

            st.write("")
            st.write("🔴 Heart Disease")
            st.progress(float(disease))
            st.write(f"{disease*100:.2f}%")

            st.write("")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Healthy Probability",
                    f"{healthy*100:.2f}%"
                )

            with c2:
                st.metric(
                    "Disease Probability",
                    f"{disease*100:.2f}%"
                )

        st.warning(
            """
                ⚠️ This prediction is generated using a Machine Learning model and should not
                replace professional medical advice or diagnosis.

                Always consult a qualified healthcare professional.
            """
        )


# =====================================
# FEATURE INFORMATION
# =====================================

st.write("")
st.divider()

st.subheader("📚 Understand the Medical Features")

with st.expander("🫀 Chest Pain Type"):

    st.markdown("""
                **ATA** – Atypical Angina

                **NAP** – Non-Anginal Pain

                **TA** – Typical Angina

                **ASY** – Asymptomatic
                """)

with st.expander("❤️ Resting ECG"):

    st.markdown("""
                **Normal** – Normal ECG

                **ST** – ST-T Wave Abnormality

                **LVH** – Left Ventricular Hypertrophy
                """)

with st.expander("🏃 Exercise Angina"):

    st.markdown("""
                **Y** → Chest pain occurs during exercise.

                **N** → No chest pain during exercise.
                """)

with st.expander("📈 ST Slope"):

    st.markdown("""
                **Up** → Generally healthier.

                **Flat** → May indicate heart disease.

                **Down** → Higher cardiovascular risk.
                """)

with st.expander("📉 Old Peak"):

    st.markdown("""
                Old Peak measures ST depression caused by exercise.

                Higher values generally indicate increased heart disease risk.
                """)