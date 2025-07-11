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
1. Database connection and initial viewing
2. Sociodemographic overview
3. Work conditions for participants
4. Mental health variables
5. Diagnosed condition prevalence rates
6. Grouping variables and conclusions

## Key results: 

- With ineffective treatment, most people struggling with mental health issues find that it interferes with their work. 
- In general, mental health issues do not seem to be a big problem in the workplace as long as they have an effective treatment plan. However, even with effective treatment, these issues are found to sometimes an issue in the workplace, thus affecting productivity a little. 
- Most survey respondents in the USA, UK, and Canada have struggled or did struggle with mental health issues from 2014-2019. 
- Having a family history and past mental health struggles seems to increase the chance of having a current mental health disorder. The contrary has been shown also, where having no family history of mental health disorders and no past disorders decreases the frequency of current mental health disorders. 

## Acknowledgements

- Dataset: https://www.kaggle.com/datasets/anth7310/mental-health-in-the-tech-industry
- Libraries: sqlite3, pandas, numpy, math, matplotlib pylab, seaborn