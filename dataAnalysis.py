# Shahriyar Mammadli
# Data Analysis script for Kaggle House Prices project
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot

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
    # List of variables to check the correlation between dependent variable
    varsToCheckCor = ["LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2",
                      "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea",
                      "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch"]
    # Check correlation against the SalePrice
    correlationInfo(trainDf, varsToCheckCor)

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

# Give the details about correlation
def correlationInfo(df, vars, scatter=False):
    for var in vars:
        # Drop NAs and 0s
        tempDf = df[df[var].notna()]
        tempDf = df[pd.to_numeric(df[var]) > 0]
        print(f"Correlation between SalePrice and {var} is {np.corrcoef(tempDf[var], tempDf['SalePrice'])[0, 1]} .")
        if scatter:
            matplotlib.pyplot.scatter( tempDf[var], tempDf['SalePrice'])
            matplotlib.pyplot.show()