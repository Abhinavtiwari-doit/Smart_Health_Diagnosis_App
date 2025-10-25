from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from .model import predict

main = Blueprint('main', __name__)

# Get feature list from Training.csv for displaying symptoms in the form
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'model_training', 'Training.csv')
FEATURES = pd.read_csv(TRAIN_PATH, nrows=1).drop('prognosis', axis=1).columns.tolist()

@main.route('/')
def home():
    # Pass the symptom list to the template
    return render_template('index.html', symptoms=FEATURES)

@main.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    symptoms = [s.strip().lower() for s in data.get("symptoms", [])]

    x = np.zeros((1, len(FEATURES)), dtype=int)
    for idx, feat in enumerate(FEATURES):
        if feat.lower() in symptoms:
            x[0, idx] = 1

    pred = predict(x)
    return jsonify({"prediction": pred})
