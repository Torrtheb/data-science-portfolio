from locust import HttpUser, task, between
import random
import json


class APIUser(HttpUser):
    wait_time = between(2, 5)

    @task
    def predict(self):
        data = {
            "id": 1,
            "gender": "Male",
            "age": 65.0,
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 75.0,
            "bmi": 28.5,
            "smoking_status": "Never smoked",
            "stroke": 0,
        }

        headers = {"Content-Type": "application/json"}

        self.client.post("/predict/", data=json.dumps(data), headers=headers)
