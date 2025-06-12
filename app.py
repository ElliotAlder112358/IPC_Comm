# Importing Libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
import pickle
from flask_basicauth import BasicAuth

# Initialize Flask app
app = Flask(__name__)

# Load models and scaler
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Ensure this file exists

# Basic Authentication config
app.config['BASIC_AUTH_USERNAME'] = 'IPC'
app.config['BASIC_AUTH_PASSWORD'] = '1PCT00L'
basic_auth = BasicAuth(app)

@app.route('/')
@basic_auth.required
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    try:
        form = request.form

        # Numeric and binary input fields
        base_features = {
            'seifa_decile': float(form.get("seifa_decile")),
            'lga_sensitivity': float(form.get("lga_sensitivity")),
            'operational_jobs': float(form.get("operational_jobs")),
            'civ': float(form.get("civ")),
            'time_in_gov': float(form.get("time_in_gov")),
            'construction_jobs': float(form.get("construction_jobs")),
            'dpe_objections': float(form.get("dpe_objections")),
            'rep_political_donation_Y': 1 if form.get("rep_political_donation") == "Y" else 0,
            'min_request_Y': 1 if form.get("min_request") == "Y" else 0,
            'council_objection_Y': 1 if form.get("council_objection") == "Y" else 0,
        }

        # One-hot encoding for Application Type
        app_type_options = ['MOD', 'Ministerial Request', 'Planning Proposal', 'SSD', 'SSD & MOD']
        for opt in app_type_options:
            base_features[f'app_type_{opt}'] = 1 if form.get("app_type") == opt else 0

        # One-hot encoding for Project Type
        proj_type_options = ['D', 'DPM', 'PH']
        for opt in proj_type_options:
            base_features[f'project_type_{opt}'] = 1 if form.get("project_type") == opt else 0

        # One-hot encoding for Development Type
        dev_type_options = [
            'Coal', 'Coal seam gas', 'Commercial', 'Education',
            'Entertainment, Tourism & Recreation', 'Food Production', 'Heritage',
            'Hospital and Health', 'Industrial', 'Infrastructure', 'Mineral - Gold',
            'Mineral - Silver', 'Miscellaneous', 'Mixed Use', 'Quarry and extraction',
            'Renewable Solar', 'Renewable Wind', 'Residential', 'Seniors Housing',
            'Waste, sewerage and resource recovery'
        ]
        for opt in dev_type_options:
            base_features[f'dev_type_{opt}'] = 1 if form.get("dev_type") == opt else 0

        # Model 1 Prediction (Commissioner Count)
        model1_features = list(model1.feature_names_in_)
        x1_df = pd.DataFrame([base_features])
        for col in model1_features:
            if col not in x1_df:
                x1_df[col] = 0
        x1_df = x1_df[model1_features]

        raw_no_comm = int(model1.predict(x1_df)[0])
        mapped_no_comm = {0: 1, 1: 2, 2: 3}.get(raw_no_comm, 1)
        base_features['no_comm'] = mapped_no_comm

        # Model 2 Prediction (Panel Hours)
        model2_features = list(model2.feature_names_in_)
        x2_df = pd.DataFrame([base_features])
        for col in model2_features:
            if col not in x2_df:
                x2_df[col] = 0
        x2_df = x2_df[model2_features]

        # Scale numerical features before model2
        numerical_features = [
            'seifa_decile', 'lga_sensitivity', 'operational_jobs', 'no_comm',
            'civ', 'time_in_gov', 'construction_jobs', 'dpe_objections'
        ]
        x2_df[numerical_features] = scaler.transform(x2_df[numerical_features])

        panel_hours = float(model2.predict(x2_df)[0])
        individual_hours = round(panel_hours, 2)

        # Generate dynamic output
        member_hours = ""
        if mapped_no_comm == 1:
            member_hours += f"Estimated Chair Hours: {individual_hours}"
        elif mapped_no_comm == 2:
            member_hours += f"Estimated Chair Hours: {individual_hours}<br>"
            member_hours += f"Estimated Member 2 Hours: {individual_hours}<br>"
            member_hours += f"Total Panel Hours: {round(individual_hours * 2, 2)}"
        elif mapped_no_comm == 3:
            member_hours += f"Estimated Chair Hours: {individual_hours}<br>"
            member_hours += f"Estimated Member 2 Hours: {individual_hours}<br>"
            member_hours += f"Estimated Member 3 Hours: {individual_hours}<br>"
            member_hours += f"Total Panel Hours: {round(individual_hours * 3, 2)}"

        return render_template(
            "index.html",
            prediction_text=f"Estimated Number of Commissioners: {mapped_no_comm}",
            prediction_text2=member_hours
        )

    except Exception as e:
        return f"Prediction error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
