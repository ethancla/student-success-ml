from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

# Create Flask app
app = Flask(__name__)

# Load the trained models
logistic_model = joblib.load('models/logistic_model.pkl')
linear_model = joblib.load('')
neural_model = joblib.load('')

# Load corresponding scalers
logistic_scaler = joblib.load('models/scaler.pkl')
linear_scaler = joblib.load('')
neural_scaler = joblib.load('')

# Define feature names
feature_names = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (enrolled)',
    'Tuition fees up to date',
    'Curricular units 2nd sem (evaluations)',
    'Age at enrollment',
    'Unemployment rate'
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
    logistic_prediction = logistic_model.predict(input_scaled)[0]
    logistic_probs = logistic_model.predict_proba(input_scaled)[0]
    classes = ['Dropout', 'Enrolled', 'Graduate']

    results['Logistic Regression'] = {
        'prediction': classes[logistic_prediction],
        'probabilities': {classes[i]: f"{prob*100:.2f}%" for i, prob in enumerate(logistic_probs)}
    }

    return render_template('results.html',
                           results=results,
                           input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)
