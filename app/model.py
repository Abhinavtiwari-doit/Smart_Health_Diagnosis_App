"""
Loads trained model & label encoder, exposes a predict() function.
"""
import joblib
import os
import numpy as np

# Construct model path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model_training')

model = joblib.load(os.path.join(MODEL_DIR, 'model.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

def predict(input_vec):
    """Predict disease label from input feature vector.
    input_vec: numpy array shape (1, N)
    Returns: string disease label
    """
    pred = model.predict(input_vec)
    return label_encoder.inverse_transform(pred)[0]
