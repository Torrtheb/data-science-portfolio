## Mental Health in the Tech Industry Exploratory Data Analysis

This project has a purpose to examine the _Mental Health in the Tech Industry_ dataset to see if mental health issues were a problem in the workplace from 2014-2019, and if these mental health issues affected workplace productivity.

## Installation: 

Python 3 is used in a Jupyter Notebook, as well as the following libraries: 
- sqlite3 
- pandas
- numpy
- math
- seaborn
- matplotlib.pylab

These can be installed using pip. 

## Data: 

The _Mental Health in the Tech Industry_ dataset used for this project was taken from Kaggle, which can be found at: https://www.kaggle.com/datasets/anth7310/mental-health-in-the-tech-industry. This includes three tables: Survey (includes years where surveys were given), Question (includes survey questions), and Answer (includes all answers to survey questions). The Question and Answer tables can be linked with the primary and foreign key: QuestionID, and the answers are found in the AnswerText column. 

## Notebook structure
- Database connection and initial viewing
- Sociodemographic overview
- Mental health variables
- Diagnosed condition prevalence rates
- Mental health condition and workplace productivity


## Key results: 

- When treatment is ineffective, most respondents with mental health issues report that their condition substantially interferes with work.
- With effective treatment, reported work interference is much lower; for many respondents, mental health issues are manageable and only mildly affect productivity.
- Across the USA, UK, and Canada, a sizable share of survey participants report current or past mental health issues between 2014–2019, underscoring that these conditions are common in tech workplaces.


## Acknowledgements

- Dataset: https://www.kaggle.com/datasets/anth7310/mental-health-in-the-tech-industry
- Libraries: sqlite3, pandas, numpy, math, matplotlib pylab, seaborn
