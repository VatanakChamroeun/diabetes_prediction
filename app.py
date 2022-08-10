from flask import Flask, render_template, request
import pickle
import numpy as np
import json

# import sklearn

# Load the Logistic Regression model
classifier = pickle.load(open('finalmodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    preg = float(data['Pregnancies'])
    glucose = float(data['Glucose'])
    bp = float(data['BloodPressure'])
    st = float(data['SkinThickness'])
    insulin = float(data['Insulin'])
    bmi = float(data['BMI'])
    dpf = float(data['DiabetesPedigreeFunction'])
    age = float(data['Age'])
    data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
    my_prediction = classifier.predict(data)

    if my_prediction == 1:
        result = "Great! You DON'T have diabetes."
    if my_prediction == 0:
        result = "Oops! You Have Diabetes"

    return app.response_class(
        response=json.dumps({
            "message": result,
            "precaution": [],
            "has_diabetes": False,
            'has_heart_disease': False,
            "has_tuberculosis": False
        }),
        status=200,
        mimetype='application/json'
    )


if __name__ == '__main__':
    app.run(debug=True)
