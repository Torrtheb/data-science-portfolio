
#  Stroke Prediction Project

The goal of this project is to correctly predictict if a person will have a stroke or not using 10 characteristics (age, BMI, average glucose level, hypertension, heart_disease, gender, ever_married, work type, residence type, and smoking status).

# Installation: 

Python 3 is used in a Jupyter Notebook with one python file for necessary functions. Additional python functions are also present for deploying the selected model. The following libraries are also used: 
- pandas
- numpy
- seaborn
- matplotlib.pylab
- scipy.stats
- phik
- scikit learn
- imblearn.ensemble
- xgboost
- lightgbm
- joblib
- FastApi (model deployment)
- Docker (model deployment)
These can be installed using pip, and Docker can be installed on the following website: https://www.docker.com. 

To access the model with Docker, 
1. Make sure that Docker is installed, and open it. 
2. Build a Docker image: docker build -t my-ml-model .
3. Make sure the image is created by running: docker images
4. Run the container: docker run -d -p 8000:8000 my-ml-model
5. Make sure the container is running: docker ps


3. Run the test_request.py file. Alternatively, send a test request using: 

curl -X 'POST' 'http://localhost:8000/predict/' \
    -H 'Content-Type: application/json' \
    -d data in json format

Either option here will return a probability that the data instance will have a stroke ranging from 0 (no stroke) to 1 (stroke). 

4. If interested, can test the model deployment performance using the Locust file by running the command: locust. Then, the http://localhost:8089 browser can be opened and the host, number of users, and spawn rate can be set. 


# Data: 

The _Stroke Prediction_ dataset can be found on Kaggle: (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), which is also added to the repository. 

# Notebook structure: 

Project Flow: 
- Introduction
- Importing necessary libraries and data
- Exploratory Data Analysis
- Statistical inference
- Model implementation
- Model deployment
- Conclusion
- References
- Appendix

# Key results: 

After a variety of models were tested, a tuned balanced random forest model is selected as the best one for predicting whether a new customer will purchase this company’s travel insurance or not. This model had accuracy and recall of 76% for new data with the area under the precision-recall curve of 0.26. This means that the model is able to accurately classify whether a new customer will purchase this company's travel insurance about 76% of the time.

Important features in predicting if a given patient will have a stroke or not were found to be their age, average blood glucose level, bmi, if they have hypertension or not, and if their bmi value was missing in the original data or not. When each of these features increase in value, then it has been shown that the probability of having a stroke increases as well. 

This balanced random forest model is of use in the health industry, as it could help to predict if an incoming patient to a hospital is in danger of having a stroke or not, and as it explains key features that can indicate an increase in stroke likelihood for a patient.

There is quite a bit of room for improvement; however, as this model does not correctly classify all  cases. More work could be done in stacking models that are good at predicting patients that have a stroke and patients that do not have a stroke (ex: logistic regression and gradient boosting). Being aided by a medical professional could also help with feature selection, model selection, and explanations for model coefficients.


# Acknowledgements

- https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmi_tbl.pdf
- https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- chatgpt