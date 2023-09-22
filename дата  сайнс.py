# Import necessary libraries

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm

# Load the dataset

arithmetic_test = pd.read_csv("arithmetic_test.csv")

# Exploratory Data Analysis

# Plotting graphs

# Graph 1: Relationship between age and mental_arithm

plt.scatter(arithmetic_test["age"], arithmetic_test["mental_arithm"])

plt.xlabel("Age")

plt.ylabel("Mental Arithmetic Score")

plt.title("Relationship between Age and Mental Arithmetic Score")

plt.show()

# Graph 2: Box plot of mental_arithm by gender

sns.boxplot(x=arithmetic_test["male"], y=arithmetic_test["mental_arithm"])

plt.xlabel("Gender")

plt.ylabel("Mental Arithmetic Score")

plt.title("Distribution of Mental Arithmetic Score by Gender")

plt.show()

# Graph 3: Bar plot of favorite subjects

subject_counts = arithmetic_test["fav_sub"].value_counts()

plt.bar(subject_counts.index, subject_counts.values)

plt.xlabel("Favorite Subject")

plt.ylabel("Count")

plt.title("Distribution of Favorite Subjects")

plt.show()

# Data preprocessing

# Identifying outliers

q1 = arithmetic_test["mental_arithm"].quantile(0.25)

q3 = arithmetic_test["mental_arithm"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr

upper_bound = q3 + 1.5 * iqr

outliers = arithmetic_test[(arithmetic_test["mental_arithm"] < lower_bound) | (arithmetic_test["mental_arithm"] > upper_bound)]

# Filling missing values

arithmetic_test["att_test_missed"].fillna(arithmetic_test["att_test_missed"].mean(), inplace=True)

# Encoding categorical variables

arithmetic_test = pd.get_dummies(arithmetic_test, columns=["fav_sub"], drop_first=True)

# Building linear regression model

X = arithmetic_test.drop(["mental_arithm", "Name", "Last name"], axis=1)

y = arithmetic_test["mental_arithm"]

model = LinearRegression()

model.fit(X, y)

# Calculating mean squared error

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)

# Formula of the linear regression model

formula = "mental_arithm = "

for i, coef in enumerate(model.coef_):

formula += f"{coef:.2f} * {X.columns[i]} + "

formula += f"{model.intercept_:.2f}"

# Summary of the linear regression model

X = sm.add_constant(X)

model_summary = sm.OLS(y, X).fit().summary()

# Checking assumptions of linear regression

# Assumption 1: Linearity

# Scatter plot of predicted values vs. residuals

plt.scatter(y_pred, model.resid)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Linearity: Predicted Values vs. Residuals")

plt.show()

# Assumption 2: Constant Variance (Homoscedasticity)

# Scatter plot of predicted values vs. residuals

plt.scatter(y_pred, model.resid)

plt.xlabel("Predicted Values")

plt.ylabel("Residuals")

plt.title("Constant Variance: Predicted Values vs. Residuals")

plt.axhline(0, color="red", linestyle="--")

plt.show()

# Assumption 3: Normality of Residuals

# Histogram of residuals

sns.histplot(model.resid, kde=True)

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.title("Normality of Residuals: Histogram")

plt.show()

