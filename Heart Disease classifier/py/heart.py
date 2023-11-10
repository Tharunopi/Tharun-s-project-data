from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

data_path = 'C:\\Users\\tharu\\PycharmProjects\\pythondfe\\heart_disease_data.csv'
data = pd.read_csv(data_path)

X = data.drop(columns=['target']).values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.load_model("heart_model.h5")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

@app.route('/')
def index():
    return render_template('input_form.html')

@app.route('/process_input', methods=['POST'])
def process_input():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    chest_pain_type = int(request.form['chest_pain_type'])
    resting_bp = int(request.form['resting_bp'])
    serum_cholesterol = int(request.form['serum_cholesterol'])
    fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
    resting_ecg = int(request.form['resting_ecg'])
    max_heart_rate = int(request.form['max_heart_rate'])
    exercise_induced_angina = int(request.form['exercise_induced_angina'])
    oldpeak = float(request.form['oldpeak'])
    slope_st_segment = int(request.form['slope_st_segment'])
    num_major_vessels = int(request.form['num_major_vessels'])
    thal = int(request.form['thal'])

    custom_input = np.array([[age, sex, chest_pain_type, resting_bp, serum_cholesterol, fasting_blood_sugar,
                              resting_ecg, max_heart_rate, exercise_induced_angina, oldpeak, slope_st_segment,
                              num_major_vessels, thal]])

    custom_input_scaled = scaler.transform(custom_input)

    predictions = model.predict(custom_input_scaled)

    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)

    if predicted_labels[0] == 0:
        result = "No Heart Disease"
    else:
        result = "Heart Disease"

    return f"The predicted result is: {result}"

if __name__ == '__main__':
    app.run(debug=True)
