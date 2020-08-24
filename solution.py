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

# EDA of the dataset
da.EDA(trainDf, testDf)
# Processing the variables that has NAs which means house does...
# ...not have this feature or object Note: GarageYrBlt indicates...
# ...the year when the garage is built but properties that does not...
# ...have garage is empty and since year is categorical variable...
# ...I filled the NAs with 'None'
featuresNATrain = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
hf.fillNANone(trainDf, featuresNATrain)
featuresNATest = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
hf.fillNANone(testDf, featuresNATest)
# Fill NAs with 'Other'
hf.fillNAOther(trainDf, ['MasVnrType', 'MasVnrArea'])
hf.fillNAOther(testDf, ['MasVnrType', 'MasVnrArea'])

# Fill NAs with most frequent value in these features
# Note: These variables have missing values only in train set
hf.fillWithMostFrequent(trainDf, ['Electrical'])

# Note: These variables have missing values only in test set
# Note: TotalBsmtSF is filled with 0 because, this house has NA for...
# ...other basement features which means it does not have a basement...
# ...and houses with no basement has TotalBsmtSF of 0, thus this...
# ...sample is also filled with 0
# Note: Other variables in this list have also same issue
hf.fillNAZero(testDf, ['TotalBsmtSF', 'GarageArea', 'GarageCars', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'BsmtHalfBath', 'BsmtFullBath'])

# Fill NAs with most frequent value in these features
# Note: These variables have missing values only in test set
hf.fillWithMostFrequent(testDf, ['KitchenQual', 'SaleType', 'Exterior1st', 'Exterior2nd', 'Functional', 'Utilities', 'MSZoning'])

# Information about missing values
da.infoMisVal(trainDf, trainDf.shape[0], "train")
da.infoMisVal(testDf, testDf.shape[0], "test")

# Drop unnecessary columns
trainDf = trainDf.drop(columns=['LotFrontage', 'Id'])
testDf = testDf.drop(columns=['LotFrontage', 'Id'])
# TODO: not handled: LotFrontage

# Merge, encode, and split again
# testDf['SalePrice'] = np.nan
mergedDf = pd.concat([trainDf, testDf], ignore_index=True)
mergedDf.to_csv("merged.csv", header=True, index=False)
import time
time.sleep(12121)
mergedDf = hf.ordinalEncoding(mergedDf)

trainDf = mergedDf[0:trainDf.shape[0]]
testDf = mergedDf[trainDf.shape[0]:mergedDf.shape[0]]

modelGB, predictionsGB = hf.gbModel(trainDf, testDf, 'SalePrice')
print(predictionsGB)
# print(hf.calcR2(actualDf['SalePrice'], predictionsGB))
# Modelling TODO: uses lasso, ridge elasticnet, SVM, RF, GB