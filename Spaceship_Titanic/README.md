
#  Project: Spaceship Titanic 

The goal of this project is to correctly predictict if a passenger aboard the _Spaceship Titanic_ will be transported to an alternate dimension or not based on a training dataset with 8693 rows and 14 features to help rescue crews efficiently find and save these missing people.

# Data: 

The _Spaceship Titanic_ dataset can be found on Kaggle: (https://www.kaggle.com/competitions/spaceship-titanic/overview), which is also added to the repository. 

# Files: 

This repository contains two jupyter notebooks with the project contents (eda_stat.ipynb and feat_eng_model.ipynb), two python files for use (func_3.py and preprocessing.py), the test, training, and submission data (.csv files), a requirements.txt file as well as a pyproject.toml file. The following libraries are also used: 
- pandas
- numpy
- seaborn
- matplotlib.pylab
- scipy.stats
- phik
- scikit learn
- xgboost
- lightgbm

# Notebook structure: 

Project Flow: 

Notebook 1: 
- Introduction
- Importing necessary libraries and data
- Exploratory Data Analysis
- Statistical inference

Notebook 2: 
- Variable preprocessing
- Model implementation
- Feature engineering
- Validation set evaluaation
- Conclusion
- References
- Appendix


# Key results: 

This project had objective of classifying passengers on the _Spaceship Titanic_ that were transported to a new dimension post dust cloud collision. This is important for safety reasons, as the passengers in an alternate dimension could be in danger. Therefore, knowing which passengers are affected by the crash in the most efficient way can help to save resources and hopefully find the missing people. 

To do this, the data was explored and statistically tested before applying various machine learning models, and tuning these to obtain the highest accuracy possible. 

After a variety of models were tested, a stacked model using histogram gradient boosted models and light gradient boosted models was selected, as it had the highest accuracy on validation data (0.808511 with the stacked model, which corresponds to a score of 0.80453 on Kaggle’s submission board). This means that the model is able to accurately classify whether a new customer will purchase this company's travel insurance about 80% of the time.

The most important features for the tuned models were found to be: Spa, CryoSleep_True or _False, tot_spend, and VRDeck using SHAP values. Notably, Spa, VRDeck, and tot_spend push the transported prediction towards 0, or towards a passenger not being transported. This means that higher spendings in the Spa, VRDeck, or in general reduce the chance of a passenger being transported. Interestingly, the FoodCourt variable shows the opposite trend, where spending more money in the FoodCourt increases chances of being transported to an alternate dimension. In addition, passengers that are in cryosleep are also transported more frequently to an alternate dimension. 

To further improve this project, an automated feature engineering using a package such as feature tools could have been examined, to further enhance feature selection. Principal component analysis or another form of automated feature selection could also have been performed prior to final model selection, to ensure that the models used the most important features only, and could have helped for model explainability. In addition, additional models could have been tuned and added to the stacked model to hopefully improve accuracy score even more. 



# Acknowledgements

- Addison Howard, Ashley Chow, and Ryan Holbrook. Spaceship Titanic. https://kaggle.com/competitions/spaceship-titanic, 2022. Kaggle.
- ChatGPT.