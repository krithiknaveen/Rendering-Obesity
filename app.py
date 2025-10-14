# -*- coding: utf-8 -*-
"""app.ipynb - Gemini API version"""

from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models and scalers
model_4_category = joblib.load('model_4_category.pkl')
model_7_category = joblib.load('model_7_category.pkl')
scaler_4_category = joblib.load('scaler_4_category.pkl')
scaler_7_category = joblib.load('scaler_7_category.pkl')

# Define columns and mappings
required_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS',
                   'Age', 'Height', 'Weight', 'BMI', 'CH2O', 'FAF', 'FCVC', 'NCP', 'TUE']

text_to_number_map = {
    'Gender': {'Male': 0, 'Female': 1},
    'family_history_with_overweight': {'No': 0, 'Yes': 1},
    'FAVC': {'No': 0, 'Yes': 1},
    'CAEC': {'No': 0, 'Some': 1, 'Full': 2},
    'SMOKE': {'No': 0, 'Yes': 1},
    'SCC': {'No': 0, 'Yes': 1},
    'CALC': {'None': 0, 'Some': 1, 'Full': 2},
    'MTRANS': {'Motorbike': 0, 'Bike': 1, 'Public': 2, 'Walking': 3}
}

def preprocess_input(data, model_type):
    for key in text_to_number_map:
        if key in data:
            data[key] = text_to_number_map[key].get(data[key], data[key])

    if 'BMI' not in data or pd.isnull(data['BMI']):
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)

    input_data = pd.DataFrame([data])
    input_data = input_data.reindex(columns=required_columns).fillna(0)

    if model_type == '4_category':
        input_data = input_data[scaler_4_category.feature_names_in_]
        input_scaled = scaler_4_category.transform(input_data)
    else:
        input_data = input_data[scaler_7_category.feature_names_in_]
        input_scaled = scaler_7_category.transform(input_data)

    return input_scaled


# ========== GEMINI FUNCTIONS ==========

def get_gemini_response(prompt, max_output_tokens=300):
    """Call Gemini model with given prompt"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        print(f"Error generating Gemini response: {str(e)}")
        return "No response generated."


def get_gpt_suggestions(prediction_label, user_data):
    prompt = f"""
    User data: {user_data}.
    Based on their BMI and predicted category '{prediction_label}', 
    provide a personalized **diet plan** in clear bullet points.
    """
    suggestions = get_gemini_response(prompt)
    return suggestions if suggestions else "No diet plan provided."


def get_gpt_suggestions1(prediction_label, user_data):
    prompt = f"""
    User data: {user_data}.
    Based on their BMI and predicted category '{prediction_label}', 
    provide a personalized **exercise plan** in clear bullet points.
    """
    suggestions = get_gemini_response(prompt)
    return suggestions if suggestions else "No exercise plan provided."


# ========== FLASK ROUTES ==========

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    model_type = request.form['model_type']
    if model_type == '4_category':
        return redirect(url_for('form_4_category'))
    else:
        return redirect(url_for('form_7_category'))

@app.route('/form_4_category')
def form_4_category():
    return render_template('form_4_category.html')

@app.route('/form_7_category')
def form_7_category():
    return render_template('form_7_category.html')

@app.route('/predict_4_category', methods=['POST'])
def predict_4_category():
    input_data = {
        'Gender': request.form['Gender'],
        'family_history_with_overweight': request.form['family_history_with_overweight'],
        'FAVC': request.form['FAVC'],
        'CAEC': request.form['CAEC'],
        'SMOKE': request.form['SMOKE'],
        'SCC': request.form['SCC'],
        'CALC': request.form['CALC'],
        'MTRANS': request.form['MTRANS'],
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'CH2O': float(request.form['CH2O']),
        'FAF': float(request.form['FAF']),
        'FCVC': float(request.form['FCVC']),
        'NCP': float(request.form['NCP']),
        'TUE': float(request.form['TUE'])
    }

    input_scaled = preprocess_input(input_data, '4_category')
    prediction = model_4_category.predict(input_scaled)
    prediction_label = ['Underweight', 'Normal', 'Obesity', 'Overweight'][prediction[0]]

    suggestion_diet = get_gpt_suggestions(prediction_label, input_data)
    suggestion_exercise = get_gpt_suggestions1(prediction_label, input_data)

    return render_template('result_4_category.html',
                           prediction_label=prediction_label,
                           suggestion_diet=suggestion_diet,
                           suggestion_exercise=suggestion_exercise)

@app.route('/predict_7_category', methods=['POST'])
def predict_7_category():
    input_data = {
        'Gender': request.form['Gender'],
        'family_history_with_overweight': request.form['family_history_with_overweight'],
        'FAVC': request.form['FAVC'],
        'CAEC': request.form['CAEC'],
        'SMOKE': request.form['SMOKE'],
        'SCC': request.form['SCC'],
        'CALC': request.form['CALC'],
        'MTRANS': request.form['MTRANS'],
        'Age': float(request.form['Age']),
        'Height': float(request.form['Height']),
        'Weight': float(request.form['Weight']),
        'CH2O': float(request.form['CH2O']),
        'FAF': float(request.form['FAF']),
        'FCVC': float(request.form['FCVC']),
        'NCP': float(request.form['NCP']),
        'TUE': float(request.form['TUE'])
    }

    input_scaled = preprocess_input(input_data, '7_category')
    prediction = model_7_category.predict(input_scaled)

    prediction_label = ['Insufficient Weight', 'Normal Weight', 'Obesity Type_I', 'Obesity Type II',
                        'Obesity Type III', 'Overweight Level I', 'Overweight Level II'][prediction[0]]

    suggestion_diet = get_gpt_suggestions(prediction_label, input_data)
    suggestion_exercise = get_gpt_suggestions1(prediction_label, input_data)

    return render_template('result_7_category.html',
                           prediction_label=prediction_label,
                           suggestion_diet=suggestion_diet,
                           suggestion_exercise=suggestion_exercise)

if __name__ == "__main__":
    app.run(debug=True)
