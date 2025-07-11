#  Project

The goal of this project is to correctly predict if a Home Credit customer will default on a loan or not, and explain key reasoning for the rejection or acceptance of a Home Credit loan application. 
This is a binary classification problem using supervised machine learning to assign probabilities of a given client defaulting on a loan for this test dataset.

The evaluation metric for this project is the area under receiver operator characteristic (ROC) curve (AUC). The ROC curve is a plot of true positive rate versus false positive rate for different thresholds. The area under this curve gives a measure of overall model performance, where a score of 1 is a perfect classifier, and a score of 0.5 is the same as random guessing. This is obtained using a model's predicted probabilities. The objective roc-auc score for this project is 0.78. 

# Data: 

The _Home Credit Default Risk_ dataset can be found on Kaggle: (https://www.kaggle.com/competitions/home-credit-default-risk/overview). 


# Files and installation: 

For data analysis, model building and testing: this repository contains 5 jupyter notebooks with the project contents (app_eda, app_model, table_eda, table_model, and test_data), one python file for use in these notebooks (functions.py), a python file (csv_to_parquet.py) used to change aggregated .csv files to smaller .parquet files, and a pyproject.toml file. The notebooks and initial data are in the notebooks_and_initial_tables folder. Saved models are in the models folder. Initial data is not included here due to its large volume, but can be found on the website shown in the Data section.

For model deployment: this repository contains two .csv and two parquet files for aggregated tables (p_final_merged and bureau_final), model .pkl files (best_lgbm_model.pkl, best_catboost.pkl, and voting.pkl), a Dockerfile, two python files: main.py for api creation, and preprocessing.py for data preprocessing, a requirements.txt file, and a test_input.json file to test deployment. All files related to initial model deployment are in the deploy folder.

For model deployment via a streamlight application, there is an app.py file, a preprocessing.py file (the same as for model deployment, essentially), the same parquet and model files as for model deployment, a requirements.txt file, and a valid_data csv file which holds data for sending test requests. All files related to streamlit deployment are in the streamlit folder. 

The following libraries are also used: 
- pandas
- numpy
- seaborn
- matplotlib.pylab
- scipy.stats
- phik
- scikit learn
- catboost
- lightgbm
- joblib
- FastApi (model deployment)
- Docker (model deployment)
- Google cloud (model deployment)
- Googld cloud storage (model deployment)
- Streamlit (model deployment)

These can be installed using pip, and Docker can be installed on the following website: https://www.docker.com. 

To send a simple test request with google cloud, a request in the form of: 
curl -X POST "https://homecredit-app-547010730225.northamerica-northeast1.run.app/predict" \
    -H "Content-Type: application/json" \
    -d @test_input.json
can be sent to the model's google cloud url: https://homecredit-app-547010730225.northamerica-northeast1.run.app, where test_input.json is a json file with test input. 

Alternatively, the streamlit app can be accessed via: https://my-streamlit-app-547010730225.northamerica-northeast1.run.app. 

To use the app, one must choose to input a specific client id or a random sample, and click the predict risk button. Note: Due to cloud deployment characteristics, the first click on the Random Client button click loads the model and data, and the second click provides the prediction results.

# Notebook structure: 

Project Flow: 

Notebook 1 (app_eda): 
- Initial dataset exploration
- Correlations
- Variable exploration
- Statistical Inference

Notebook 2 (app_model) : 
- Splitting data into validation and training sets, 
- Data preprocessing, 
- Model preparation, 
- Model evaluation, 
- Feature Importance examination, 
- Feature engineering, 
- Final preprocessing selection

Notebook 3 (table_eda) : 
For each table, 
- Initial dataset exploration
- Correlations
- Variable exploration

Notebook 4 (table_model): 
- Data preprocessing
- Feature engineering 
- Table merging
- Model preparation and initial evaluation
- Feature importance and selection
- Hyperparameter tuning
- Model ensembles
- Testing models on holdout data
- Confusion matrix examination and model selection
- Feature importance of selected model

Notebook 5 (test_data): 
- Merging tables
- Preprocessing data
- Model preparation
- Model evaluation
- Conclusion and improvements 


# Key results: 

This project had objective of classifying Home Credit customers applying for loans. This is quite important to ensure that Home Credit as a profitable company, as it needs to minimize the chance that any loans given out will not be repayed. To do this, the data was explored and statistically tested before applying various machine learning gradient boosted models, and tuning these to obtain the highest accuracy possible. 

Next, a light-gb, a catboost, a hist-gb, and a voting model were trained and implemented to predict the default probability of Home Credit's customers. The best performing model was found to be the voting model, which takes and votes on predictions from the three boosted models. This got a performance of 0.792 for roc-auc for the validation dataset, which is the main evaluation metric. When this model was evaluated with the test dataset on Kaggle's platform, it obtained a score of 0.78879. 

However, the fastest and more explainable model is light-gb, with roc-auc score of 0.790 for the validation dataset, and 0.78709 for the test dataset on Kaggle's platform. This light-gb model was chosen for deployment, as speed and confidence in model reasoning are deemed more important than a slight decrease in performance for Home Credit's organization. Both of these roc-auc scores are above 0.78, which is the objective score for this project. 

From looking at the SHAP summary plot for this model, key observations include: 

- External credit rating scores are by far the most important in determining the default probability of a customer. In each instance, it has been seen in SHAP summary plots that having high external credit scores lead to a lower likelihood of a customer defaulting on a loan. 
- Sociodemographic factors also influence a customer’s likelihood of defaulting on their Home Credit loan, notably: the length of their employment, age, gender, marriage status, if they own a car, and if they have higher education or not. 
- Finally, economic factors and repayment tendencies describing the customer’s loan influence their default probability, notably: the annuity on their current loan application, the credit/annuity ratio of their current loan application, the amount of debt they had on previous loans at the credit bureau, or their late payment behaviour on previous Home Credit loans. 

Project improvements include 
- The models could be trained with more recent data, as this Kaggle competition is 8 years old. This would to keep the model relevant and up to date and ensure that Home Credit continues to provide their loaning services to the maximum amount of trustworthy customers. 
- In addition, infrastructure for more thorough testing and monitoring for model deployment could be implemented, to ensure that the model's performance and input data stay constant over time. 
- It would also be interesting to have trained other (simpler) model types, such as a logistic regression model with a smaller number of features. As long as obtained roc-auc score does not drop significantly with the simpler model, Home Credit would be able to pinpoint exactly why a customer's loan application is rejected or not, and by exactly how much each feature influences the acceptance decision. 
- Threshold tuning or model calibrated could have been performed to test model reliability.
- Finally, adding in a more user-friendly and speedy interface would be useful to the streamlit application webpage. 



# Acknowledgements
- https://www.homecredit.net
- https://en.wikipedia.org/wiki/Home_Credit
- Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.
- https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821
- ChatGPT.
