# Shahriyar Mammadli
# Helper functions of a Kaggle House Prices project's solution script
# Import required libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

# This function puts 'None' in place of NAs in specific variables
def fillNANone(df, vars):
    for var in vars:
        df[var] = df[var].fillna("None")
    return df

# Fill the missing value with most frequent value in the variable
def fillWithMostFrequent(df, vars):
    for var in vars:
        df[var] = df[var].fillna(df[var].mode()[0])
    return df

# Fill with 'Other'
def fillNAOther(df, vars):
    for var in vars:
        df[var] = df[var].fillna('Other')
    return df

# Fill with 0
def fillNAZero(df, vars):
    for var in vars:
        df[var] = df[var].fillna(0)
    return df

# Ordinal Encoding
def ordinalEncoding(df, vars):
    for var in vars:
        df[var] = LabelEncoder().fit_transform(df[var])
    return df

# Random Forest Model
def rfModel(trainDf, testDf, targetVar):
    rf = RandomForestRegressor()
    # Fit data
    rf = rf.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return rf, rf.predict(testDf.drop(targetVar, 1))

# Gradient Boosting Model
def gbModel(trainDf, testDf, targetVar):
    gb = GradientBoostingRegressor()
    # Fit data
    gb = gb.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return gb, gb.predict(testDf.drop(targetVar, 1))

# Calculate MAPE
def MAPE(actual, predicted):
    return (np.mean(np.abs((actual - predicted) / actual)) * 100).round(1)
