from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__, template_folder='template')

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route("/result", methods=['POST', 'GET'])
def result():
    try:
        # Correct the handling of gender input
        gender_str = request.form['gender'].lower()
        if gender_str not in ['male', 'female']:
            raise ValueError("Invalid gender value")

        # Convert gender to numeric value if needed
        gender = 1 if gender_str == 'male' else 0

        # Remaining input processing code...
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                      avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

        scaler_path = os.path.join('G:\STROKE PROBABILITY DETECTOR\Models\scaler.pkl')
        scaler = None
        with open(scaler_path, 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        x = scaler.transform(x)

        model_path = os.path.join('G:\STROKE PROBABILITY DETECTOR\Models\dt.sav')
        dt = joblib.load(model_path)

        Y_pred = dt.predict(x)

        # for No Stroke Risk
        if Y_pred == 0:
            return render_template('nostroke.html')
        else:
            return render_template('stroke.html')

    except ValueError as e:
        return render_template('error.html', message=str(e))


if __name__ == "__main__":
    app.run(debug=True, port=7384)
