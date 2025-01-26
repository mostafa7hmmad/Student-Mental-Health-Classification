# Student Mental Health Classification App
![img](img.jpg)
The Student Mental Health Classification App is an interactive web-based tool designed to assess mental health conditions among students. This application leverages machine learning to predict whether a student might be experiencing symptoms of depression based on various lifestyle, academic, and mental health factors.

The app aims to create awareness about mental health and provide insights to encourage early intervention. It is built using Streamlit and deployed on Streamlit Community Cloud, making it accessible to users online without requiring them to log in.

## Key Features

### User Input Interface:
Users can input their data through sliders and dropdowns for various features such as:
- **Age**
- **Academic Pressure**
- **Study Satisfaction**
- **Dietary Habits**
- **Suicidal Thoughts**
- **Financial Stress**
- **Study Hours Per Day**

### Machine Learning Predictions:
- The app uses a pre-trained **Random Forest Classifier** to predict the likelihood of depression.
- Based on the prediction:
  - If depression is likely, the app displays a message encouraging the user to seek professional help.
  - If depression is unlikely, the app provides positive reinforcement.

### Performance Evaluation:
- Users can upload a test dataset (CSV) to evaluate the performance of the Random Forest model, with metrics such as:
  - **Classification Report**
  - **Confusion Matrix**

### Interactive Animation:
- A visually appealing JavaScript animation reinforces the app's theme: **"Mental Health Matters"**.

### Fully Hosted Online:
- The app is deployed on **Streamlit Community Cloud** and publicly accessible via a custom URL.

## Technologies Used

### Frontend:
- **Streamlit**

### Backend:
- **Python**

### Machine Learning Model:
- **Random Forest Classifier** (pre-trained with `joblib`)

### Libraries:
- **pandas**, **numpy**: For data handling and preprocessing.
- **joblib**: To load the pre-trained model.
- **scikit-learn**: For metrics like the classification report.
- **matplotlib**: For visualizations (optional use case).

## How It Works

### Input Data:
- Users provide their data via an intuitive sidebar form.
- Features include demographic, academic, and lifestyle factors.

### Model Prediction:
- The input data is processed and fed into the Random Forest Classifier.
- The app predicts whether the user is at risk of depression.

### Result Display:
- The app provides actionable insights based on the model's prediction.
- Encouraging messages are displayed to promote mental health awareness.

### Performance Testing:
- Users can upload a CSV file to test the model's performance on a sample dataset.

## Target Audience

- **Students** looking to assess their mental health.
- **Educational Institutions** to raise awareness about mental health among their student communities.
- **Mental Health Professionals** seeking tools for awareness campaigns.

## Purpose and Impact

The app aims to:

1. **Encourage Early Intervention:** By providing insights into mental health conditions, it promotes seeking professional help early.
2. **Raise Awareness:** The app educates users about factors influencing mental health.
3. **Promote Positivity:** Through motivational messages and support.

