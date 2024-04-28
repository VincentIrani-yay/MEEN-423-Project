import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Load data from the CSV file
df = pd.read_csv("star_classification.csv")

# Convert categories to numbers and splitting into features and response
df["class"]=[0 if i == "GALAXY" else 1 if i == "STAR" else 2 for i in df["class"]]

features = df.loc[:, df.columns != "class"]
response = df['class']

# Splitting test and training data
x_train, x_test, y_train, y_test = train_test_split(features, response, test_size=0.2, random_state = 53)

# Scaling the data
scaler = StandardScaler()
scaler.fit(x_train)
x_train_sc = scaler.transform(x_train)
x_test_sc = scaler.transform(x_test)


