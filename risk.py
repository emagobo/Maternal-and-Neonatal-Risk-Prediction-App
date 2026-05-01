import pandas as pd
from streamlit_option_menu import option_menu
import streamlit as st
import joblib

# MODELS
scaler = joblib.load('maternal_scaler.pkl')
model_rf = joblib.load('maternal_model.pkl')

scaler_n = joblib.load('neonatal_scaler.pkl')
model_gb = joblib.load('neonatal_model.pkl')


# GUI Creation - PAGE SETUP
st.set_page_config(page_title='Maternal & Fetal Risk Predictor', layout='wide', page_icon='')
st.title('Maternal and Neonatal Risk Prediction System')

st.markdown("<p style='font-weight:bold;'>About</p>", unsafe_allow_html=True)
st.write(
    'This app predict likelyhood of maternal risk level during pregnacy and fetal health assessment based on the clinical data'
    'A Maternal and Fetal Risk Prediction System structured around maternal, fetal, and newborn assessment provides a continuous framework for identifying health risks across pregnancy and early life.')


st.caption('®Developed by Dr. Elisha Magobo | @2026 | Machine Learning Project | Maternal Project')

# SIDEBAR
with st.sidebar:
    selected = option_menu('Maternal and Fetal Health Prediction',
                           ['Maternal Health Assessment', 'Neonatal Health Assessment'],
                           menu_icon='hospital-fill', icons=['activity', 'heart', 'person'], default_index=0)

# =========================
# MATERNAL HEALTH PAGE
# =========================

if selected == 'Maternal Health Assessment':

    st.header('Maternal Risk Prediction')

    st.write("Enter patient clinical parameters below:")

    col1, col2 = st.columns(2)

    with col1:

        Age = st.number_input(
            'Age of the Pregnant mother',
            18, 70, 30
        )

        Systolic_BP = st.number_input(
            'Systolic Blood Pressure(mmHg)',
            90, 160, 120
        )

        Diastolic_BP = st.number_input(
            'Diastolic Blood Pressure(mmHg)',
            60, 100, 90
        )

        Blood_Sugar = st.number_input(
            'Blood Sugar Level(mg/dL)',
            5.0, 28.0, 15.6
        )

        Body_Temp = st.number_input(
            'Body Temperature (˚C)',
            50.0, 135.0, 100.0
        )

        BMI = st.number_input(
            'Body Mass Index',
            10.0, 25.0, 18.0
        )

    with col2:

        Previous_Complications = st.selectbox(
            'History of previous Complications',
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        Preexisting_Diabetes = st.selectbox(
            'Preexisting Diabetes',
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        Gestational_Diabetes = st.selectbox(
            'Gestational Diabetes',
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        Mental_Health = st.selectbox(
            'Mental Health Condition',
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        Heart_Rate = st.number_input(
            'Maximum Heart Rate',
            60.0, 220.0, 135.0
        )

    # =========================
    # FEATURE BUILDER
    # =========================

    def build_maternal_features():

        return pd.DataFrame([{

            "Age": Age,
            "Systolic_BP": Systolic_BP,
            "Diastolic_BP": Diastolic_BP,
            "Blood_Sugar": Blood_Sugar,
            "Body_Temp": Body_Temp,
            "BMI": BMI,
            "Previous_Complications": Previous_Complications,
            "Preexisting_Diabetes": Preexisting_Diabetes,
            "Gestational_Diabetes": Gestational_Diabetes,
            "Mental_Health": Mental_Health,
            "Heart_Rate": Heart_Rate

        }])

    # =========================
    # PREDICTION BUTTON
    # =========================

    if st.button("Predict Risk", key="predict_maternal"):

        try:

            # Build features
            features = build_maternal_features()

            # Predict
            prediction = model_rf.predict(features)[0]

            probability = model_rf.predict_proba(features)[0]

            # Convert probabilities to percentages
            probabilities = {
                "Low Risk": round(probability[0] * 100, 2),
                "High Risk": round(probability[1] * 100, 2)
            }

            # =========================
            # DISPLAY RESULTS
            # =========================

            st.subheader("Prediction Result")

            if prediction == 0:

                st.success("✅ Low Risk")

            else:

                st.warning("⚠️ High Risk")

            # =========================
            # DISPLAY PROBABILITIES
            # =========================

            st.subheader("Prediction Probabilities")

            st.write(
                f"Low Risk: {probabilities['Low Risk']:.2f}%"
            )

            st.write(
                f"High Risk: {probabilities['High Risk']:.2f}%"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


# ===============
# NEONATAL HEALTH PAGE
# ===============

if selected == 'Neonatal Health Assessment':
    st.header('Neonatal Risk Prediction')
    st.write("Enter Neonate clinical parameters below:")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender of the Baby:', ['Male', 'Female'])
        gestational_age_weeks = st.number_input('Gestational age at birth in Weeks', 20.0, 42.0, 35.0)
        birth_weight_kg = st.number_input('Birth Weight in Kg:', 0.5, 5.0, 3.2)
        birth_length_cm = st.number_input('Length at Birth in cm:', 30.0, 60.0, 35.9)
        birth_head_circumference_cm = st.number_input('Head circumference at birth in cm:', 20, 40, 33)
        age_days = st.number_input('Age of Baby in days since Birth:', 0, 60, 12)
        weight_kg = st.number_input('Daily Updated weight (gram/day):', 20, 35, 22)

    with col2:
        length_cm = st.number_input("Fetal Length (cm)", value=50.0)
        head_circumference_cm = st.number_input('Daily updated head circumference:', 30, 45, 30)
        temperature_c = st.number_input('Body temperature in °C', 30.0, 41.0, 36.8)
        heart_rate_bpm = st.number_input('Heart Rate:', 90, 160, 120)
        respiratory_rate_bpm = st.number_input('Breathing Rate (breaths/min):', 30, 70, 55)
        oxygen_saturation = st.number_input('SpO₂ level (%):', 50, 100, 95)
        feeding_type = st.selectbox('Feeding Type:', ['Breastfeeding', 'Formula', 'Mixed'])

    with col3:
        feeding_frequency_per_day = st.number_input('Number of feeds per day:', 7, 12, 8)
        urine_output_count = st.number_input('Wet Diapers/day:', 5, 15, 10)
        stool_count = st.number_input('Bowel Movements per day:', 0, 8, 4)
        jaundice_level_mg_dl = st.number_input('Bilirubin Level (mg/dL):', 1, 25, 15)
        apgar_score = st.number_input('APGAR Score at Birth:', 0, 10, 8)
        immunizations_done = st.selectbox('Immunizaition done (BCG, HepB, OPV on Day 1 & 30):', ['Yes', 'No'])
        reflexes_normal = st.selectbox('Newborn Reflex:', ['Yes', 'No'])


    def build_neonatal_features():
        # -----------------------------
        # Encoding maps (CRITICAL)
        # -----------------------------
        gender_map = {"Male": 0, "Female": 1}

        feeding_map = {
            "Breastfeeding": 0,
            "Formula": 1,
            "Mixed": 2
        }

        binary_map = {
            "No": 0,
            "Yes": 1
        }

        data = {
            "gender": gender_map[gender],
            "gestational_age_weeks": float(gestational_age_weeks),
            "birth_weight_kg": float(birth_weight_kg),
            "birth_length_cm": float(birth_length_cm),
            "birth_head_circumference_cm": float(birth_head_circumference_cm),
            "age_days": float(age_days),
            "weight_kg": float(weight_kg),
            "length_cm": float(length_cm),
            "head_circumference_cm": float(head_circumference_cm),
            "temperature_c": float(temperature_c),
            "heart_rate_bpm": float(heart_rate_bpm),
            "respiratory_rate_bpm": float(respiratory_rate_bpm),
            "oxygen_saturation": float(oxygen_saturation),
            "feeding_type": feeding_map[feeding_type],
            "feeding_frequency_per_day": float(feeding_frequency_per_day),
            "urine_output_count": float(urine_output_count),
            "stool_count": float(stool_count),
            "jaundice_level_mg_dl": float(jaundice_level_mg_dl),
            "apgar_score": float(apgar_score),
            "immunizations_done": binary_map[immunizations_done],
            "reflexes_normal": binary_map[reflexes_normal]
        }

        df = pd.DataFrame([data])

        # Force numeric → fails early if anything is wrong
        df = df.astype(float)

        return df


    if st.button("Predict Newborn Risk", key="predict_newborn"):

        try:
            features = build_neonatal_features()

            prediction = model_gb.predict(features)[0]
            probability = model_gb.predict_proba(features)[0]

            # Convert to percentages
            probabilities = {
                "Healthy": round(probability[0] * 100, 2),
                "At Risk": round(probability[1] * 100, 2)
            }

            st.subheader("Prediction Newborn Result")
            if prediction == 0:

                st.success(
                    "✅ Healthy: All newborn vitals normal"
                )

            elif prediction == 1:

                st.warning(
                    "⚠️ Newborn at Risk: Mild jaundice, slight fever, "
                    "SpO₂ 92–95%. Refer to pediatrician."
                )

            st.subheader("Prediction Probabilities")

            st.write(
                f"Healthy: {probabilities['Healthy']:.2f}%"
            )

            st.write(
                f"At Risk: {probabilities['At Risk']:.2f}%"
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")


