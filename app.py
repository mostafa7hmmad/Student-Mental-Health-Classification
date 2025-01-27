import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from streamlit.components.v1 import html

# Function to render the initial cube animation with 6 cubes
def render_js_animation():
    animation_code = """
    <div class="container" style="position:relative; height:300px; display:flex; justify-content:center; align-items:center;">
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
        <div class="flex">
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
            <div class="cube">
                <div class="wall front"></div>
                <div class="wall back"></div>
                <div class="wall left"></div>
                <div class="wall right"></div>
                <div class="wall top"></div>
                <div class="wall bottom"></div>
            </div>
        </div>
    </div>
    <style>
        .flex {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 10px;
        }

        .cube {
            position: relative;
            width: 60px;
            height: 60px;
            transform-style: preserve-3d;
            animation: rotate 4s infinite;
            margin: 10px;
        }

        .wall {
            position: absolute;
            width: 60px;
            height: 60px;
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid #ddd;
        }

        .front { transform: translateZ(30px); }
        .back { transform: translateZ(-30px) rotateY(180deg); }
        .left { transform: rotateY(-90deg) translateX(-30px); transform-origin: center left; }
        .right { transform: rotateY(90deg) translateX(30px); transform-origin: center right; }
        .top { transform: rotateX(90deg) translateY(-30px); transform-origin: top center; }
        .bottom { transform: rotateX(-90deg) translateY(30px); }

        @keyframes rotate {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            50% { transform: rotateX(180deg) rotateY(180deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }
    </style>
    """
    html(animation_code, height=300)

# Function to render result animation
def render_result_animation(result):
    if result == 1:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#ff5722;animation:bounce 1s infinite;"></div>
            <style>
                @keyframes bounce {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-20px); }
                }
            </style>
        </div>
        """
    else:
        animation_code = """
        <div style="display:flex;justify-content:center;align-items:center;height:200px;">
            <div style="width:100px;height:100px;border-radius:50%;background-color:#4caf50;animation:scale 1s infinite;"></div>
            <style>
                @keyframes scale {
                    0%, 100% { transform: scale(1); }
                    50% { transform: scale(1.2); }
                }
            </style>
        </div>
        """
    html(animation_code, height=200)

# Function to create the user input form
def centered_form():
    with st.form(key="user_input_form"):
        st.markdown("### Enter Your Details")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            age = st.text_input("Age", "25")
            academic_pressure = st.selectbox(
                "Academic Pressure (1-5)",
                options=[
                    "1: Very low (minimal workload, no stress)",
                    "2: Low (manageable workload, minor stress)",
                    "3: Moderate (average workload, some stress)",
                    "4: High (heavy workload, considerable stress)",
                    "5: Very high (extremely demanding workload, intense stress)"
                ],
                index=2
            )
            study_satisfaction = st.selectbox(
                "Study Satisfaction (1-5)",
                options=[
                    "1: Very dissatisfied (completely unhappy)",
                    "2: Dissatisfied (struggling to enjoy studies)",
                    "3: Neutral (neither satisfied nor dissatisfied)",
                    "4: Satisfied (mostly happy and motivated)",
                    "5: Very satisfied (completely fulfilled and enjoying studies)"
                ],
                index=2
            )
            dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

        with col2:
            suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
            study_hours = st.text_input("Study Hours per Day", "4")
            financial_stress = st.selectbox(
                "Financial Stress (1-5)",
                options=[
                    "1: No stress (completely stable finances)",
                    "2: Low stress (minor financial concerns)",
                    "3: Moderate stress (some financial difficulties)",
                    "4: High stress (frequent financial worries)",
                    "5: Very high stress (severe financial burden)"
                ],
                index=2
            )

        marital_status = st.selectbox("Marital Status", ["Yes", "No"])
        anxiety = st.selectbox("Do you have Anxiety?", ["Yes", "No"])
        panic_attack = st.selectbox("Do you have Panic Attack?", ["Yes", "No"])
        specialist_treatment = st.selectbox("Did you seek any specialist for a treatment?", ["Yes", "No"])

        # Map categorical inputs to numerical values
        dietary_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
        binary_map = {"Yes": 1, "No": 0}

        # Extract numerical values from user-friendly selectbox strings
        academic_pressure_value = int(academic_pressure.split(":")[0])
        study_satisfaction_value = int(study_satisfaction.split(":")[0])
        financial_stress_value = int(financial_stress.split(":")[0])

        rf_data = pd.DataFrame({
            "Age": [int(age)],
            "Academic Pressure": [academic_pressure_value],
            "Study Satisfaction": [study_satisfaction_value],
            "Dietary Habits": [dietary_map[dietary_habits]],
            "Have you ever had suicidal thoughts?": [binary_map[suicidal_thoughts]],
            "Study Hours": [int(study_hours)],
            "Financial Stress": [financial_stress_value]
        })

        nb_lr_data = pd.DataFrame({
            "Marital status": [binary_map[marital_status]],
            "Do you have Anxiety?": [binary_map[anxiety]],
            "Do you have Panic attack?": [binary_map[panic_attack]],
            "Did you seek any specialist for a treatment?": [binary_map[specialist_treatment]]
        })

        submit_button = st.form_submit_button(label="Predict Mental Health Condition")

    return rf_data, nb_lr_data, submit_button

def main():
    st.set_page_config(page_title="Mental Health Prediction", layout="wide")

    st.markdown("""<div style="text-align:center;">
    <h3>🌟 You are not alone, and your mental health matters! 🌟</h3>
    <p>This tool uses machine learning models to provide insights into mental health conditions. Remember, this is not a substitute for professional help. 💙</p>
    </div>""", unsafe_allow_html=True)

    # Render the cube animation
    render_js_animation()

    # Render the centered form
    rf_data, nb_lr_data, submit_button = centered_form()

    if submit_button:
        st.write("Processing... Please wait.")
        with st.spinner("Predicting..."):
            time.sleep(2)

            # Load models
            rf_model = joblib.load("best_rf_model.pkl")
            nb_model = joblib.load("best_nb.pkl")
            logistic_model = joblib.load("beat_logistic.pkl")

            # Random Forest prediction
            rf_prediction = rf_model.predict(rf_data.values)[0]

            # Naive Bayes prediction
            nb_prediction = nb_model.predict(nb_lr_data.values)[0]

            # Logistic Regression prediction
            logistic_prediction = logistic_model.predict(nb_lr_data.values)[0]

            # Voting mechanism
            predictions = [rf_prediction, nb_prediction, logistic_prediction]
            final_prediction = max(set(predictions), key=predictions.count)

            # Display results
            st.subheader("Prediction Results")
            if final_prediction == 1:
                st.markdown(
                    "#### The system detects a potential mental health condition. Please consult a professional for further guidance."
                )
                render_result_animation(1)
            else:
                st.markdown(
                    "#### No mental health condition detected. Keep maintaining a healthy lifestyle!"
                )
                render_result_animation(0)

if __name__ == "__main__":
    main()
