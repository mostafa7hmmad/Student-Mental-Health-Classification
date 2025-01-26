
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from streamlit.components.v1 import html

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
        .bottom { transform: rotateX(-90deg) translateY(30px); transform-origin: bottom center; }

        @keyframes rotate {
            0% { transform: rotateX(0deg) rotateY(0deg); }
            50% { transform: rotateX(180deg) rotateY(180deg); }
            100% { transform: rotateX(360deg) rotateY(360deg); }
        }
    </style>
    """
    html(animation_code, height=300)

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

def main():
    st.set_page_config(page_title="Mental Health Prediction", layout="wide")

    # Header
    st.title("Mental Health Prediction App")
    render_js_animation()

    st.markdown("---")
    st.markdown("### 🌟 You are not alone, and your mental health matters! 🌟")
    st.markdown(
        "#### This tool uses machine learning models to provide insights into mental health conditions. Remember, this is not a substitute for professional help. 💙"
    )

    # Sidebar
    st.sidebar.header("User Input Features")

    def user_input_features():
        def validate_input(value, min_val, max_val, field_name):
            try:
                value = int(value)
                if value < min_val or value > max_val:
                    st.sidebar.error(f"{field_name} must be between {min_val} and {max_val}.")
                    return None
                return value
            except ValueError:
                st.sidebar.error(f"{field_name} must be an integer.")
                return None

        age = validate_input(st.sidebar.text_input("Age", "25"), 10, 60, "Age")
        academic_pressure = validate_input(st.sidebar.text_input("Academic Pressure (1-5)", "3"), 1, 5, "Academic Pressure")
        study_satisfaction = validate_input(st.sidebar.text_input("Study Satisfaction (1-5)", "3"), 1, 5, "Study Satisfaction")
        financial_stress = validate_input(st.sidebar.text_input("Financial Stress (1-5)", "3"), 1, 5, "Financial Stress")
        study_hours = validate_input(st.sidebar.text_input("Study Hours per Day (1-16)", "4"), 1, 16, "Study Hours")

        dietary_habits = st.sidebar.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        suicidal_thoughts = st.sidebar.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])

        # Ensure all inputs are valid before returning data
        if None in [age, academic_pressure, study_satisfaction, financial_stress, study_hours]:
            return None

        # Map inputs to numerical values
        dietary_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
        suicidal_map = {"No": 0, "Yes": 1}

        data = {
            "Age": age,
            "Academic Pressure": academic_pressure,
            "Study Satisfaction": study_satisfaction,
            "Dietary Habits": dietary_map[dietary_habits],
            "Have you ever had suicidal thoughts?": suicidal_map[suicidal_thoughts],
            "Financial Stress": financial_stress,
            "Study Hours": study_hours,
        }
        return pd.DataFrame(data, index=[0])

    df_input = user_input_features()

    if df_input is not None:
        # Display user input
        st.subheader("User Input Features")
        st.write(df_input)

        # Load Random Forest model
        model = joblib.load("best_rf_model.pkl")

        # Predict using the model
        if st.button("Predict Mental Health Condition"):
            st.write("Processing... Please wait.")
            with st.spinner("Predicting... This tool provides insights but is not a substitute for professional advice."):
                time.sleep(2)  # Simulate processing time

                # Prepare input
                X_input = df_input.values

                try:
                    prediction = model.predict(X_input)[0]  # Binary prediction (0 or 1)
                    prediction_text = (
                        "Yes (Mental health condition detected)"
                        if prediction == 1
                        else "No (No mental health condition detected)"
                    )

                    # Display results
                    st.subheader("Prediction Results")

                    message = """
                    <div style="background-color:#e3f2fd;padding:20px;margin:20px;border-radius:15px;box-shadow:0 4px 8px rgba(0,0,0,0.1);">
                        <h3 style="color:#1565c0;">Prediction Result</h3>
                        <p style="font-size:18px;color:#333;">The prediction is: <b style='color:{};'>{}</b></p>
                        <p style="font-size:16px;color:{};">{}</p>
                    </div>
                    """.format(
                        "#ff5722" if prediction == 1 else "#4caf50",
                        prediction_text,
                        "#ff5722" if prediction == 1 else "#4caf50",
                        "We encourage you to consult a mental health professional for further guidance." if prediction == 1 else "Your mental health looks positive. Keep maintaining healthy habits!",
                    )
                    st.markdown(message, unsafe_allow_html=True)

                    # Render result animation
                    render_result_animation(prediction)

                except Exception as e:
                    st.error(f"Error during prediction: {e}")

    # Batch prediction section
    st.markdown("---")
    st.subheader("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV File for Batch Prediction", type="csv")

    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.write(df_batch.head())

            # Ensure necessary columns exist
            required_columns = ["Age", "Academic Pressure", "Study Satisfaction", "Dietary Habits", "Have you ever had suicidal thoughts?", "Financial Stress", "Study Hours"]
            if all(col in df_batch.columns for col in required_columns):
                # Map dietary habits and suicidal thoughts columns
                dietary_map = {"Healthy": 0, "Moderate": 1, "Unhealthy": 2}
                suicidal_map = {"No": 0, "Yes": 1}

                df_batch["Dietary Habits"] = df_batch["Dietary Habits"].map(dietary_map)
                df_batch["Have you ever had suicidal thoughts?"] = df_batch["Have you ever had suicidal thoughts?"].map(suicidal_map)

                X_batch = df_batch[required_columns].values
                predictions = model.predict(X_batch)
                df_batch["Prediction"] = ["Yes" if pred == 1 else "No" for pred in predictions]

                st.write("Prediction Results:")
                st.write(df_batch)

                # Option to download results
                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")

            else:
                st.error(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Footer
    st.markdown("---")
    st.markdown("*Developed for mental health awareness and prediction.*")

if __name__ == "__main__":
    main()
