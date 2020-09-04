# Shahriyar Mammadli
# Solution of Kaggle House Prices competition
# Import required libraries
import pandas as pd
import dataAnalysis as da
import helperFunctions as hf
import numpy as np

# Reading files
trainDf = pd.read_csv('../Datasets/KaggleHousePrices/train.csv')
testDf = pd.read_csv('../Datasets/KaggleHousePrices/test.csv')
originalDf = testDf
# EDA of the dataset
da.EDA(trainDf, testDf)

# Processing the variables that has NAs which means house does...
# ...not have this feature or object Note: GarageYrBlt indicates...
# ...the year when the garage is built but properties that does not...
# ...have garage is empty and since year is categorical variable...
# ...I filled the NAs of these properties with 'None'
featuresNATrain = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
                   'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure',
                   'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
trainDf = hf.fillNANone(trainDf, featuresNATrain)
featuresNATest = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType',
                  'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure',
                  'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
testDf = hf.fillNANone(testDf, featuresNATest)

# Fill NAs with 'Other'
trainDf = hf.fillNAOther(trainDf, ['MasVnrType'])
testDf = hf.fillNAOther(testDf, ['MasVnrType'])
trainDf = hf.fillNAZero(trainDf, ['MasVnrArea'])
testDf = hf.fillNAZero(testDf, ['MasVnrArea'])

# Fill NAs with most frequent value in these features
# Note: These variables have missing values only in train set
trainDf = hf.fillWithMostFrequent(trainDf, ['Electrical'])

# Note: These variables have missing values only in test set
# Note: TotalBsmtSF is filled with 0 because, this house has NA for...
# ...other basement features which means it does not have a basement...
# ...and houses with no basement has TotalBsmtSF of 0, thus this...
# ...sample is also filled with 0
# Note: Other variables in this list have also same issue
testDf = hf.fillNAZero(testDf, ['TotalBsmtSF', 'GarageArea', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtHalfBath', 'BsmtFullBath'])

# Fill NAs with most frequent value in these features
# Note: These variables have missing values only in test set
testDf = hf.fillWithMostFrequent(testDf, ['KitchenQual', 'SaleType', 'Exterior1st', 'Exterior2nd', 'Functional', 'Utilities', 'MSZoning'])

# Information about missing values
da.infoMisVal(trainDf, trainDf.shape[0], "train")
da.infoMisVal(testDf, testDf.shape[0], "test")

# Drop unnecessary columns
trainDf = trainDf.drop(columns=['Id'])
testDf = testDf.drop(columns=['Id'])

# Encode the dataframe
trainDf, testDf = hf.encodeDF(trainDf, testDf)
# Handle LotFrontage
trainDf["LotFrontage"], testDf["LotFrontage"] = hf.regressVar(trainDf, testDf, "LotFrontage")
# The variables outliers are dropped as a result of analyzing the scatter plot
# Scatter plotting code is int dataAnalysis.py script, send scatter=True as...
# ...an input to the correlationInfo() function to enable it
# trainDf = trainDf[trainDf['LotFrontage'].astype(np.float64) < 300]
# trainDf = trainDf[trainDf['LotArea'].astype(np.float64) < 200000]
# trainDf = trainDf[trainDf['MasVnrArea'].astype(np.float64) < 1200]
# trainDf = trainDf[trainDf['BsmtFinSF1'].astype(np.float64) < 4000]
# trainDf = trainDf[trainDf['BsmtFinSF2'].astype(np.float64) < 1400]
# trainDf = trainDf[trainDf['TotalBsmtSF'].astype(np.float64) < 6000]
# trainDf = trainDf[trainDf['1stFlrSF'].astype(np.float64) < 4000]
# trainDf = trainDf[trainDf['GrLivArea'].astype(np.float64) < 5000]
# trainDf = trainDf[trainDf['GarageArea'].astype(np.float64) < 1200]
# trainDf = trainDf[trainDf['WoodDeckSF'].astype(np.float64) < 800]
# trainDf = trainDf[trainDf['OpenPorchSF'].astype(np.float64) < 500]
# trainDf = trainDf[trainDf['EnclosedPorch'].astype(np.float64) < 500]
# trainRows, trainCols = trainDf.shape
# print(f"Train Data {trainRows} rows and {trainCols} columns")
# print("-----------------------------------------------------------")

# Perform min-max scaling
varsToScale = ['BsmtFinSF2', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
# trainDf, testDf = hf.minmaxScaling(trainDf, testDf, varsToScale)

trainDf, testDf = hf.procesingVars(trainDf, testDf)

# Build a XGBoost Model
modelXG, predictionsXG = hf.xgbModel(trainDf, testDf, 'SalePrice')
predictionsXG = np.exp(predictionsXG)
print(f"R-squared of the model for in-sample data is {(100*modelXG.score(trainDf.drop('SalePrice', 1), trainDf['SalePrice'])).round(1)}%")
featureDf = pd.DataFrame({'columns': trainDf.drop('SalePrice', 1).columns, 'importances': modelXG.feature_importances_})
# featureDf.to_csv("impotances.csv", header=True, index=False)
# Create a submission file
pd.DataFrame({'Id': originalDf['Id'], 'SalePrice': predictionsXG}).to_csv("submission.csv", header=True, index=False)
