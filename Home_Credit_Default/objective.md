# Capstone Project 

## Context: 

Home Credit is a financial lending company that was founded in 1997 in the Czech Republic. This company's main purpose is to help people with no or very little credit history to safely borrow money. This can be in the form of cash, through an online app, through credit cards, or through product loans. Home Credit's website indicates that their lending process is simple and easy, and they have expanded over the past 25 years to other countries in Asia or Europe, such as the Philippines, Indonesia, China, or Slovakia. Their customers are approved for loans using transactional and telco information. To ensure the longevity and success of Home Credit, it is important to accurately predict if a customer will default on a loan or not before they are approved. Defaulting on a loan means that a customer has not repaid their loan in the agreed timeline with Home Credit. 


## General assumptions: 
- Data is accurate, 
- Samples are independant, and 
- the distribution of training data is the same as the distribution of test data. 


## Project objectives: 

This is a binary classification problem using supervised machine learning to assign probabilities of a given client defaulting on a loan for this test dataset. Therefore, the key objective of this project is to predict which Home Credit clients are capable of loan repayment. In addition, identifying and explaining factors that influence loan repayment is essential. 

The evaluation metric for this project is the area under receiver operator characteristic (ROC) curve (AUC). The ROC curve is a plot of true positive rate versus false positive rate for different thresholds. The area under this curve gives a measure of overall model performance, where a score of 1 is a perfect classifier, and a score of 0.5 is the same as random guessing. This is obtained using a model's predicted probabilities. 

Objectives for each project part are defined: 

## Exploratory Data Analysis

Objective: gather preliminary insights form data. More specifically, this part aims to: 
    - Explore the dataset and examine relationships between tables in this dataset. 
    - Find important correlations between target and other variables.
    - Look into patterns between variables.
    - Find and address any sources or risks of data leakage.
    - Find and address missing values, outliers, or duplicates. 

Assumptions: The key assumption for this part of the project is that the application_test and application_train data follow the same distribution. As the target variable is available in the application_test dataset alone, variables are explored with the training set only. 

## Statistical Inference

Objective: elucidate correlations between the target variable and other important variables using statistical tests. In addition, this part aims to identify statistically significant features for the prediction of default risk. 

Assumptions: The data comes from a random and representative sample of Home Credit's customer population. In addition, is is assumed that samples are independent of each other.  

## Machine learning model implementation

Objective: Train and select a machine learning model to predict if a Home Credit customer will default or not on a loan, using area under ROC curve as the evaluation metric. The objective roc-auc (area under the receiver operator curve) score for this project is 0.78. 

Assumptions: With a dataset this large, it is assumed that best performing models will also need to be quite efficient. Gradient boosted models are selected here, as they can handle missing values on their own, and their performance is generally fast. In addition, it is assumed that no data leakage happens throughout the model training process, that feature distributions are constant through time, and that the data is representative of Home Credit's customer population. Another key assumption is that a significant relationship exists between the given data and the target variable. 

## Model deployment

Objective: provide access and implementation for the chosen best model, so that Home Credit can accurately predict default risk for their clients. 

Assumptions: It is assumed that the production data will have the same structure and distribution as training data, and that any engineered features or preprocessing techniques will retain their use and meaning. In addition, the model is thought to generalize well to new data, and will run efficiently in Google Cloud's environment. Another key assumption is that no private information about any of Home Credit's customers will be leaked.  


## References: 
- https://www.homecredit.net
- https://en.wikipedia.org/wiki/Home_Credit
- Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.
- https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821
