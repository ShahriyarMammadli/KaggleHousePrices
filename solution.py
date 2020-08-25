# Shahriyar Mammadli
# Solution of Kaggle House Prices competition
# Import required libraries
import pandas as pd
import dataAnalysis as da
import helperFunctions as hf

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
# ...I filled the NAs with 'None'
featuresNATrain = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
trainDf = hf.fillNANone(trainDf, featuresNATrain)
featuresNATest = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'GarageYrBlt']
testDf = hf.fillNANone(testDf, featuresNATest)

# Fill NAs with 'Other'
# TODO: build a model to predict MasVnrType and MasVnrArea
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
trainDf = trainDf.drop(columns=['LotFrontage', 'Id'])
testDf = testDf.drop(columns=['LotFrontage', 'Id'])
# TODO: not handled: LotFrontage

# Merge, encode, and split again
mergedDf = pd.concat([trainDf, testDf], ignore_index=True)
mergedDf = mergedDf.astype(str)
# These variables are ordinal encoded since they are non-numeric
# Note: GarageYrBlt is numerical its NA values are replaced with...
# ...'Other' and since it is a categorical values(years) it is also...
# ...ordinal encoded
varsToEncode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterCond', 'MasVnrType', 'ExterQual',
                'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'GarageYrBlt']
mergedDf = hf.ordinalEncoding(mergedDf, varsToEncode)
trainDf = mergedDf[0:trainDf.shape[0]]
testDf = mergedDf[trainDf.shape[0]:mergedDf.shape[0]]
# mergedDf.to_csv("mergedDf.csv", header=True, index=False)

# Build a Gradient Boosting Model
modelGB, predictionsGB = hf.gbModel(trainDf, testDf, 'SalePrice')
# Create a submission file
pd.DataFrame({'Id': originalDf['Id'], 'SalePrice': predictionsGB}).to_csv("submission.csv", header=True, index=False)

print(f"R-squared of the model for in-sample data is {(100*modelGB.score(trainDf.drop('SalePrice', 1), trainDf['SalePrice'])).round(1)}%")
# pd.DataFrame({'columns': trainDf.drop('SalePrice', 1).columns, 'importances': modelGB.feature_importances_}).to_csv("impotances.csv", header=True, index=False)

# Build a Random Forest Model
# modelRF, predictionsRF = hf.rfModel(trainDf, testDf, 'SalePrice')
# # Create a submission file
# pd.DataFrame({'Id': originalDf['Id'], 'SalePrice': predictionsRF}).to_csv("submission.csv", header=True, index=False)
#
# print(f"R-squared of the model for in-sample data is {(100*modelRF.score(trainDf.drop('SalePrice', 1), trainDf['SalePrice'])).round(1)}%")
# # pd.DataFrame({'columns': trainDf.drop('SalePrice', 1).columns, 'importances': modelRF.feature_importances_}).to_csv("impotances.csv", header=True, index=False)

# TODO: uses lasso, ridge elasticnet, SVM, RF, GB

# TODO: hypertune