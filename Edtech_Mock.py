#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 21:12:08 2022

@author: venky
"""

import pandas as pd ## Data manipulation
import numpy as np ## numerical(mathematical) calculations

## Load the data
edt = pd.read_csv("/home/venky/Desktop/Datascience_360/Real_Project_costprediction/MOCK_DATA-dup.csv")

## Understanding the data
edt.info()  ## information about the null,data type, memory
x = edt.describe() ## statistical information
edt.shape ## (1000, 18)
edt.columns
"""['s.no', 'Institute', 'Subject', 'Location', 'Trainer_Qualification',
       'Online_classes', 'Offline_classes', 'Trainer_experiance',
       'Course_level', 'Course_hours', 'Course_rating', 'Rental_permises',
       'Trainer_slary', 'Maintaince_cost', 'Non_teaching_staff_salary',
       'Placements', 'Certificate', 'Price']"""

edt.drop(['s.no'], axis = 1, inplace = True)
edt.shape  ## (1000, 17)

## Data cleaning

# Type casting

# Duplicates
edt.duplicated().sum()  ## no duplicates

# null values
edt.isna().sum()  ## no null values

# Outliers

cols = [ 'Trainer_experiance', 'Course_hours', 'Course_rating', 'Rental_permises',
       'Trainer_slary', 'Maintaince_cost', 'Non_teaching_staff_salary', 'Price']

## check ing outliers
import seaborn as sns
import matplotlib.pyplot as plt

for i in cols:
    sns.boxplot(edt[i]); plt.show() 
## we have no outliers(Random generation numbers)


## all columns boxplot in one graph
bx = sns.boxplot(data = edt, orient = "h") ## no outliers


## Label encoder
cols = ['Institute', 'Subject', 'Location', 'Trainer_Qualification','Online_classes', 'Offline_classes',
        'Course_level','Placements', 'Certificate']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Instantiate the encoders
encoders = {column: le for column in cols}

for column in cols:
    edt[column] = encoders[column].fit_transform(edt[column])

edt.var() ## Certificate has zero variance

edt.drop(['Certificate'], axis = 1, inplace = True)

edt.shape  ## (1000, 16)

"""### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(edt)
"""
## Exploratory Data Analysis (EDA)

# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)


# Graphical Representation

## bar plot
plt.bar(height = edt.Price, x = np.arange(1, 1001, 1))

plt.hist(edt.Price) #histogram
## Symmetrical distributed
plt.boxplot(edt.Price) #boxplot
## no outliers

# column R&D Spend
plt.bar(height = edt.Course_hours, x = np.arange(1, 1001, 1))
plt.hist(edt.Course_hours) #histogram
## right skewed
plt.boxplot(edt.Course_hours) #boxplot
## no outliers
sns.scatterplot(edt.Price, edt.Trainer_slary)
# Jointplot

## get the scatterplot and histogram in one plot
sns.jointplot(x = edt.Price, y = edt.Course_hours)

## correlation coeffecient for 2 columns
np.corrcoef(edt.Price, edt.Trainer_slary)
## r = 0.02188

# Countplot ( i didn't get the any information)
plt.figure(1, figsize = (16, 10))
sns.countplot(edt.Price)  ## frequency of the values

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(edt.Maintaince_cost, dist = "norm", plot = pylab) ## values are normally distributed
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(edt.iloc[:, :])  ## all combinations in one plot
## or
sns.pairplot(edt,  hue = 'Course_hours', palette = "Set1")
                             
# Correlation matrix 
corr = edt.corr()
plt.figure(1, figsize=(16, 10))
sns.heatmap(corr, annot = True, fmt = '.2%')


edt.columns

import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Institute + Subject + Location + Trainer_Qualification + Online_classes + Offline_classes + Course_level + Course_hours + Course_rating + Rental_permises + Trainer_slary + Maintaince_cost + Non_teaching_staff_salary + Placements', data = edt).fit() # regression model


# Summary
ml1.summary()
# p-values for all columns are less than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals

# Check for Colinearity to decide to remove a variable using VIF (variance influcence factor)
# Assumption: VIF > 10 = colinearity exist :VIF= (1/1-R^2)
# calculating VIF's values of independent variables

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = edt[list(edt.select_dtypes(include = ['int64', 'float64']).columns)]

# Profit feature is dependent or out put feature so we are deleting
X = X.drop('Price', axis = 1)

## VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

## calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]
print(vif_data)

## all columns are less than 10 vif(no collinearity).


## Creating the model

from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, make_scorer

rmse =lambda y, y_hat: np.sqrt(mean_squared_error(y, y_hat))

x_train, x_test, y_train, y_test = train_test_split(edt.drop("Price", axis = 1), edt.Price, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(x_train, y_train)

lm_preds = lm.predict(x_test)
rmse(y_test, lm_preds)

rmse_scorer = make_scorer(rmse, greater_is_better = False)
pipeline_optimizer = TPOTRegressor(
    scoring = rmse_scorer,
    max_time_mins = 10,
    random_state = 42,
    verbosity = 2
    )
pipeline_optimizer.fit(x_train, y_train)

print(pipeline_optimizer.score(x_test, y_test))

pipeline_optimizer.fitted_pipeline_


pipeline_optimizer.export('Mock2.py')


from sklearn.ensemble import AdaBoostRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

exported_pipeline = make_pipeline(
    Nystroem(gamma=0.6000000000000001, kernel="linear", n_components=1),
    AdaBoostRegressor(learning_rate=0.5, loss="linear", n_estimators=100)
)

exported_pipeline.fit(x_train, y_train)

y_pred_test = exported_pipeline.predict(x_test)

result_test = pd.DataFrame({'Actual':y_test, "Predicted": y_pred_test})
result_test.head(10)

## importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# predicting the accuracy score
score_test = r2_score(y_test, y_pred_test)

print('R2 score(test): ', score_test)
print('Mean squared error(test): ', mean_squared_error(y_test, y_pred_test))
print('Root Mean squared error(test): ', np.sqrt(mean_squared_error(y_test, y_pred_test)))

""" 
R2 score(test):  0.9597868664997279
Mean squared error(test):  4301.207761251101
Root Mean squared error(test):  65.58359368966526
"""

y_pred_train = exported_pipeline.predict(x_train)

result_train = pd.DataFrame({'Actual':y_train, "Predicted": y_pred_train})
result_train.head(10)

# predicting the accuracy score
score_train = r2_score(y_train, y_pred_train)

print('R2 score(train): ', score_train)
print('Mean squared error(train): ', mean_squared_error(y_train, y_pred_train))
print('Root Mean squared error(train): ', np.sqrt(mean_squared_error(y_train, y_pred_train)))

"""
R2 score(train):  0.9753400195191615
Mean squared error(train):  3011.3292841336506
Root Mean squared error(train):  54.87558003459873
"""

plt.scatter(y_test, y_pred_test)






















