import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
from streamlit.components.v1 import html

# Load models
best_rf = joblib.load("best_rf_model.pkl")

# JavaScript Animation
def render_js_animation():
    animation_code = """
    <div id="animation" style="width:100%;height:200px;display:flex;justify-content:center;align-items:center;background-color:#f0f8ff;border-radius:10px;">
        <h2 style="color:#0073e6;font-family:Arial, sans-serif;">✨ Mental Health Matters ✨</h2>
    </div>
    <script>
        const animationDiv = document.getElementById('animation');
        let colors = ["#f28c28", "#4caf50", "#2196f3", "#e91e63"];
        let i = 0;
        setInterval(() => {
            animationDiv.style.backgroundColor = colors[i % colors.length];
            i++;
        }, 1000);
    </script>
    """
    html(animation_code, height=200)

# Streamlit App
def main():
    st.set_page_config(page_title="Mental Health Prediction", layout="wide")

    # Header
    st.title("Mental Health Prediction App")
    render_js_animation()

    st.markdown("---")
    st.markdown("### 🌟 You are not alone, and your mental health matters! 🌟")
    st.markdown(
        "#### This tool is designed to provide insights into mental health conditions and support awareness. Please remember, seeking professional help is always encouraged. 💙"
    )

    # Sidebar
    st.sidebar.header("User Input Features")

    def user_input_features():
        age = st.sidebar.slider("Age", 10, 60, 25)
        academic_pressure = st.sidebar.slider("Academic Pressure (1-5)", 1, 5, 3, step=1)
        study_satisfaction = st.sidebar.slider("Study Satisfaction (1-5)", 1, 5, 3, step=1)
        dietary_habits = st.sidebar.slider("Dietary Habits (0-2)", 0, 2, 1, step=1)
        suicidal_thoughts = st.sidebar.selectbox("Have you ever had suicidal thoughts?", [0, 1])
        financial_stress = st.sidebar.slider("Financial Stress (1-5)", 1, 5, 3, step=1)
        study_hours = st.sidebar.slider("Study Hours per Day (1-16)", 1, 16, 4, step=1)

        data = {
            "Age": age,
            "Academic Pressure": academic_pressure,
            "Study Satisfaction": study_satisfaction,
            "Dietary Habits": dietary_habits,
            "Have you ever had suicidal thoughts?": suicidal_thoughts,
            "Financial Stress": financial_stress,
            "Study Hours": study_hours,
        }
        return pd.DataFrame(data, index=[0])

    df_input = user_input_features()

    # Display user input
    st.subheader("User Input Features")
    st.write(df_input)

    # Predict using the models
    if st.button("Predict Mental Health Condition"):
        st.write("Processing... Please wait.")
        with st.spinner("Predicting... This tool provides insights but is not a substitute for professional advice. If you feel overwhelmed, please reach out to a mental health professional."):
            time.sleep(2)  # Simulate processing time

            # Prepare input
            X_input = df_input.values

            # Make predictions
            prediction_rf = best_rf.predict(X_input)[0]

            # Display results
            if prediction_rf:
                st.error("Our analysis suggests a high likelihood of depression. Please consider reaching out to a mental health professional for guidance and support.")
            else:
                st.success("Our analysis suggests a low likelihood of depression. Stay positive and maintain healthy habits!")

    # Add performance evaluation
    st.subheader("Model Performance")
    st.write("Evaluate models using your test data.")

    uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
        X_test = df_test.drop("Depression", axis=1).values
        y_test = df_test["Depression"].values

        y_pred_rf = best_rf.predict(X_test)

        st.write("Classification Reports:")

        st.text("Random Forest:")
        st.text(classification_report(y_test, y_pred_rf))

    # Footer
    st.markdown("---")
    st.markdown("*Developed for mental health awareness and prediction.*")

if __name__ == "__main__":
    main()
