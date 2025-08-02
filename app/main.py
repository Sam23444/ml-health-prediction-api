from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models from the models/ folder relative to this file
with open(os.path.join('app', 'models', 'diabetes_model.pkl'), 'rb') as f:
    diabetes_model = pickle.load(f)

with open(os.path.join('app', 'models', 'heart_model.pkl'), 'rb') as f:
    heart_model = pickle.load(f)


@app.route('/')
def home():
    return "Welcome to the Health Prediction API"


@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    expected_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]

    data = request.get_json()

    # Check for missing fields
    missing = [field for field in expected_fields if field not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        # Convert inputs to correct types
        input_data = [float(data[field]) for field in expected_fields]
        input_array = np.array([input_data])
        prediction = diabetes_model.predict(input_array)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    data = request.get_json()

    try:
        # Required feature columns
        expected_columns = [
            'age', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar',
            'resting_ecg_results', 'max_heart_rate', 'exercise_induced_angina',
            'depression_induced_by_exercise', 'slope_of_st_segment',
            'number_of_major_vessels', 'thalassemia', 'chest_pain_type'
        ]

        # Check for missing fields
        for col in expected_columns:
            if col not in data:
                return jsonify({'error': f'Missing field: {col}'}), 400

        # Put into DataFrame
        input_df = pd.DataFrame([data])[expected_columns]

        # Let the model pipeline handle preprocessing
        prediction = heart_model.predict(input_df)[0]
        result = "At Risk of Heart Disease" if prediction == 1 else "No Heart Disease"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
