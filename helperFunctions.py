# Shahriyar Mammadli
# Helper functions of a Kaggle House Prices project's solution script
# Import required libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# This function puts 'None' in place of NAs in specific variables
def fillNANone(df, vars):
    for var in vars:
        df[var] = df.fillna("None", inplace=True)
    return df

# Fill the missing value with most frequent value in the variable
def fillWithMostFrequent(df, vars):
    for var in vars:
        df[var] = df.fillna(df[var].mode()[0], inplace=True)
    return df

# Fill with 'Other'
def fillNAOther(df, vars):
    for var in vars:
        df[var] = df.fillna('Other', inplace=True)
    return df

# Fill with 0
def fillNAZero(df, vars):
    for var in vars:
        df[var] = df.fillna(0, inplace=True)
    return df

# Ordinal Encoding
def ordinalEncoding(df):
    return df.apply(LabelEncoder().fit_transform)

# Random Forest Model
def rfModel(trainDf, testDf, targetVar):
    rf = RandomForestRegressor(n_estimators=100)
    # Fit data
    rf = rf.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return rf, rf.predict(testDf.drop(targetVar, 1))

# Gradient Boosting Model
def gbModel(trainDf, testDf, targetVar):
    gb = GradientBoostingRegressor(n_estimators=100, max_features='sqrt')
    # Fit data
    gb = gb.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return gb, gb.predict(testDf.drop(targetVar, 1))

# Calculate R-squared
def calcR2(actual, predicted):
    return r2_score(actual, predicted)

# Calculate MAPE
def calcR2(actual, predicted):
    return mean_absolute_error(actual, predicted)
