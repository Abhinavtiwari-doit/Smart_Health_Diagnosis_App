import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Load training data
train = pd.read_csv('Training.csv', low_memory=False)

# Encode target labels
label_encoder = LabelEncoder()
train['prognosis'] = label_encoder.fit_transform(train['prognosis'])

# Separate features and target
X = train.drop('prognosis', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
y = train['prognosis']

# You can split some training data into validation for evaluation or use test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set and calculate accuracy
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# Save the model and label encoder
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Model and label encoder saved.")
