from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

# Create Flask app
app = Flask(__name__)

# Load the trained models
logistic_model = joblib.load('')
linear_model = joblib.load('')
nueral_model = joblib.load('')

# Load corresponding scalers
logistic_scaler = joblib.load('')
linear_scaler = joblib.load('')
neural_scaler = joblib.load('')

# Define feature names
feature_names = [
    'Marital Status', 'Application mode', 'Application order', 'Course', 
    'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)', 
    "Mother's qualification", "Mother's occupation", 'Admission grade', 
    'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 
    'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 
    'Unemployment rate', 'Inflation rate', 'GDP'
]

@app.route('/')
def home():
    return render_template('index.html')

# Process input data, returns results page with predictions
# Receives student data via POST request
@app.route('predict', methods=['POST'])
def predict():
    # Get values from form
    input_data = {}
    for feature in feature_names:
        value = request.form.get(feature)
        try:
            input_data[feature] = float(value)
        except ValueError:
            return render_template('index.html', error=f"Invalid input for {feature}.")

    # Create DataFram with user inputs
    input_df = pd.DataFrame([input_data])

    # Get predictions
    results = {}

    # Logistic Regression
    input_scaled = logistic_scaler.transform(input_df)
    logistic_prediction = logistic_model.predict(input_scaled)
    logistic_probs = logistic_model.predict_proba(input_scaled)
    classes = ['Dropout', 'Enrolled', 'Graduate']

    results['Logistic Regression'] = {
        'Prediction': classes[logistic_prediction],
        'Probability': {classes[i]: f"{prob*100:.2f}%" for i, prob in enumerate(logistic_probs)}
    }

    return render_template('results.html',
                           results=results,
                           input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
