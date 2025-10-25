# Smart Health Diagnosis App

## Description
Smart Health Diagnosis App is an AI-powered web application that predicts diseases based on user-input symptoms. It combines machine learning with an intuitive web interface to provide users with quick and accurate health insights.

## Features
- Enter symptoms via a clean web interface
- Disease prediction using a Random Forest classifier
- Fast AI-powered results
- Deployable on cloud platforms for global access

## Installation
1. Clone the repository:
git clone https://github.com/Abhinavtiwari-doit/Smart_Health_Diagnosis_App.git

2. Navigate to the directory:
cd Smart_Health_Diagnosis_App

3. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

4. Install dependencies:
pip install -r requirements.txt

5. Run the app:
python run.py

6. Access locally at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Built With
Python, Flask, scikit-learn, Pandas, NumPy, joblib, Gunicorn, Render

## Model Training
Data preprocessing and model training scripts are under `model_training/`. Run `train_model.py` to train the model and generate files used in the app.

## How It Works
Users input symptoms, which are converted to feature vectors and fed into the Random Forest model to predict possible diseases. The backend uses Flask, serving predictions via API.

## Main Algorithm
Random Forest classifier was chosen due to its robustness, accuracy on symptom-disease data, and ease of integration with scikit-learn.

## Challenges
Handling data inconsistencies, model generalization, and environment differences between local and cloud deployment.

## Future Work
- NLP-based symptom input parsing
- Expanded disease dataset
- Nearby hospital finder feature

## License
MIT License

## Demo Video Youtube Link
[Demo Video](https://youtu.be/JCWXngYPGp4)