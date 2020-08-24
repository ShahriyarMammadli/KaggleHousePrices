# Shahriyar Mammadli
# Data Analysis script for Kaggle House Prices project
# Import required libraries
import pandas as pd

# EDA for dataset
def EDA(trainDf, testDf):
    trainRows, trainCols = trainDf.shape
    print(f"Train Data {trainRows} rows and {trainCols} columns")
    print("-----------------------------------------------------------")
    testRows, testCols = testDf.shape
    print(f"Test Data {testRows} rows and {testCols} columns")
    print("-----------------------------------------------------------")
    # Change printing settings of pandas to show all rows
    pd.set_option('display.max_rows', trainCols)
    # Show features which have NA samples in descending order
    infoMisVal(trainDf, trainRows, "train")
    # Show features which have NA samples in descending order
    infoMisVal(testDf, testRows, "test")
    # Print number of unique
    infoUniVal(trainDf)

# Gives an information about missing values
def infoMisVal(df, count, setName):
    print(f"Percent of missing value in the {setName} set")
    print((df.isna().sum()[df.isna().sum() > 0].sort_values()/count * 100).round(1).astype(str) + '%')
    print("-----------------------------------------------------------")

# Print number of unique
def infoUniVal(df):
    print("Number of unique values")
    print(df.nunique())
    print("-----------------------------------------------------------")