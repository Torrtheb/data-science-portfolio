from flask import Flask, jsonify, request
import requests


url = "http://localhost:8000/predict/"
data = {
    "id": 56669,
    "gender": "Male",
    "age": 81.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 186.21,
    "bmi": None,
    "smoking_status": "formerly smoked",
    "stroke": 1,
}


response = requests.post(url, json=data)
print(response.json())
