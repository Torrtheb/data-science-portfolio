# A/B Testing Project

This project has the goal of performing A/B testing on two data sets from Kaggle: _Fast Food Marketing Campaign A/B Test dataset_ and _Cookie Cats A/B Test dataset_. 

More specifically, the Fast Food Marketing Campaign notebook looks at the weekly sales over four weeks for three different marketing campaigns to select the most lucrative marketing campaign. The Cookie Cats dataset uses A/B testing to evaluate if the first gate in the _Cookie Cats_ game should be moved to level 40 or kept at level 30. 

# Installation: 

Python 3 is used in two different Jupyter Notebooks (cc.ipynb, ff.ipynb) and one python file (functions.py). Libraries can be installed using: uv pip install -r requirements.txt, which installs numpy, pandas, scipy, stastmodels, matplotlib, seaborn, ipykernel, and jupyter. 


# Data: 

The _Fast Food Marketing Campaign A/B Test dataset_ and the _Cookie Cats A/B Test dataset_ can be found on Kaggle: https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test, and https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats, respectively. These are both added to the repository as well. 

# Notebook structure: 

Each notebook (ff.ipynb, cc.ipynb) is structured in the same way: 
- Introduction
- Importing necessary libraries
- Data loading
- Exploratory Data Analysis 
- Assumptions Checking
- A/B Testing
- Decision

# Key results: 

For the Fast Food Marketing Campaign, four non-parametric tests were performed, as the distribution is not normal: 
- Bootstrapping for means (Promotion 2 was found to have the lowest total sales)
- Bootstrapping for medians (Promotion 1 was found to have the highest median total sales, with promotion 3 next)
- Kolmogorov-Smirnov Test (Promotion 1 was found to have the sales distribution shifted towards the highest values, followed by promotion 3 and then promotion 2.)

Decision for the _Fast Food Marketing Campaign A/B Test dataset_: Promotion 1 should be used in future marketing campaigns for this new menu item. 

For the Cookie Cats Game, four non-parametric tests were also performed, as the distribution was not found to be normally distributed either: 
- Bootstrapping for means (no significant difference was found)
- Bootstrapping for medians (no significant difference was found)
- Kolmogorov-Smirnov Test (for the number of gamerounds played, the version of the game with the first gate at level 30 was shifted towards higher values than the one with the first gate at level 40), though examining the cumulative distribution function plot did not show a meaningful difference between the two groups. Therefore, this conclusion shows that the difference between the two groups might be statistically significant, but very small. 
- Chi Square Test (Used for player retention after one week. This showed that the version with the first gate at level 30 had slightly better retention than the version with the first gate at level 40). 

Decision for the _Cookie Cats dataset_: The first gate in the game should be kept at level 30 to maximize player retention as well as the number of game rounds played after one week. It is important to note that a sample ratio mismatch was seen here, so these results should be interpreted with caution. 



# Acknowledgements

- Datasets: 
--https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test
--https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats
- Libraries: pandas, numpy, seaborn, matplotlib.pylab, scipy.stats, statsmodels.api