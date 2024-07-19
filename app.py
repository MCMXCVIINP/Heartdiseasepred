import os
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(BASE_DIR, 'model.pkl')
model = joblib.load(model_path)  # Ensure this path is correct

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    chestPainType = int(request.form['chestPainType'])
    restingBP = int(request.form['restingBP'])
    cholesterol = int(request.form['cholesterol'])
    fastingBS = int(request.form['fastingBS'])
    restingECG = int(request.form['restingECG'])
    maxHR = int(request.form['maxHR'])
    exerciseAngina = int(request.form['exerciseAngina'])
    oldpeak = float(request.form['oldpeak'])
    st_slope = int(request.form['st_slope'])
    
    # Create feature array
    features = np.array([[age, sex, chestPainType, restingBP, cholesterol, fastingBS, restingECG, maxHR, exerciseAngina, oldpeak, st_slope]])
    
    # Make prediction
    prediction_proba = model.predict_proba(features)[0][1] * 100  # Get the probability of heart disease
    
    # Prepare result based on the probability
    if prediction_proba < 50:
        result = "No risk of heart disease."
    else:
        result = f"You have a {prediction_proba:.2f}% chance of having heart disease."
    
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

