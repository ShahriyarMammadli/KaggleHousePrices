# Shahriyar Mammadli
# Helper functions of a Kaggle House Prices project's solution script
# Import required libraries
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import svm
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Build a regressor to fill the missing values
def regressVar(trainDf, testDf, varName):
    tempDf = trainDf.drop(columns=['SalePrice'])
    tempDf = tempDf[tempDf[varName].notna()]
    X_train, X_test, y_train, y_test = train_test_split(tempDf.drop(columns=[varName]),
                                                        tempDf[varName], test_size=0.3, random_state=42)
    X_train[varName] = y_train
    X_test[varName] = y_test
    model, predictions = rfModel(X_train, X_test, varName)
    print(f"Regression Mape value is {MAPE(y_test, predictions)} for {varName}")
    print(f"Regression R-squared value is {(100*model.score(X_test.drop(varName, 1), X_test[varName])).round(1)}%")
    return trainDf.apply(lambda i: fillNAUsingModel(i, model, varName) if pd.isnull(i[varName]) else i, axis=1)[varName],\
           testDf.apply(lambda i: fillNAUsingModel(i, model, varName) if pd.isnull(i[varName]) else i, axis=1)[varName]

# Function to encode the dataframe
def encodeDF(trainDf, testDf):
    # Merge, encode, and split again
    mergedDf = pd.concat([trainDf, testDf], ignore_index=True)
    # Note: GarageYrBlt is numerical its NA values are replaced with...
    # ...'Other' and since it is a categorical values(years) it is also...
    # ...ordinal encoded
    # These categorical variables are ordinal encoded since their categories are...
    # ...not independent of each other and have mathematical relations
    varsToEncode = ['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
                    'Fence', 'PoolQC']
    mergedDf[varsToEncode] = mergedDf[varsToEncode].astype(str)
    varsToEncode = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
                    'GarageCond', 'PoolQC']
    scaleMap1 = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap1)

    varsToEncode = ['LotShape']
    scaleMap2 = {'None': 0, 'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap2)

    varsToEncode = ['Utilities']
    scaleMap3 = {'None': 0, 'NoSeWa': 1, 'NoSewr': 2, 'AllPub': 3}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap3)

    varsToEncode = ['LandSlope']
    scaleMap4 = {'None': 0, 'Sev': 1, 'Mod': 2, 'Gtl': 3}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap4)

    varsToEncode = ['BsmtExposure']
    scaleMap5 = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3,  'Gd': 4}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap5)

    varsToEncode = ['BsmtFinType1', 'BsmtFinType2']
    scaleMap6 = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap6)

    varsToEncode = ['Fence']
    scaleMap7 = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    mergedDf = ordinalEncoding(mergedDf, varsToEncode, scaleMap7)
    # These variables categorical and they their instances are independent of each other...
    # ...thus, they are one-hot encoded
    varsToEncode = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
                    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                    'Heating', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'MoSold',
                    'SaleType', 'SaleCondition', 'Street', 'CentralAir', "GarageYrBlt"]
    mergedDf[varsToEncode] = mergedDf[varsToEncode].astype(str)
    mergedDf = oneHotEncoding(mergedDf, varsToEncode)
    # After encoding split the dataframe into train and test again
    trainDf = mergedDf[0:trainDf.shape[0]]
    testDf = mergedDf[trainDf.shape[0]:mergedDf.shape[0]]
    return trainDf, testDf

# XB Boost Model
def xgbModel(trainDf, testDf, targetVar):
    # Tuning the parameters of XGBoost
    # parameters_for_testing = {
    #     'colsample_bytree': [0.6, 0.7, 0.8],
    #     'gamma': [0, 0.03],
    #     'min_child_weight': [1.5, 6],
    #     'learning_rate': [0.01, 0.03],
    #     'max_depth': range(5, 7, 1),
    #     'n_estimators': [1000, 3000],
    #     'subsample': [0.5, 0.75],
    #     'reg_alpha': [0.75],
    #     'reg_lambda': [0.4],
    #     'seed': [42]
    # }
    # gsearch1 = GridSearchCV(estimator=XGBRegressor(), param_grid=parameters_for_testing, n_jobs=4, verbose=100, scoring='neg_mean_squared_error')
    # gsearch1.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    # print("scores")
    # print(gsearch1.best_estimator_)
    # Best parameters fot the Regressor
    model_params = {
        "max_depth": 5,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "subsample": 0.75,
        "colsample_bytree": 0.6,
        "min_child_weight": 1.5,
        "reg_alpha": 0.75,
        "reg_lambda": 0.4,
        "seed": 42
    }
    # Create train, test, and validation sets from the training data to tune the model
    X_train, X_test, y_train, y_test = train_test_split(trainDf.drop(targetVar, 1), trainDf[targetVar],
                                                        test_size=0.25, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    # Validation parameters
    val_params = {
        "eval_metric": "rmse",
        "early_stopping_rounds": 500,
        "verbose": 100,
        "eval_set": [(X_val, y_val)]
    }
    xg = XGBRegressor(**model_params)
    xg = xg.fit(X_train, y_train, **val_params)
    print(f"RMSE of the model is {mean_squared_error(y_test, xg.predict(X_test), squared=False)}")
    return xg, xg.predict(testDf.drop(targetVar, 1))

# This function does log transformation to get rid of outlier problem and...
# ...new features are created
def procesingVars(trainDf, testDf):
    # New features
    trainDf['Area'] = trainDf['LotArea'] * trainDf['LotFrontage']
    trainDf["TotalLA"] = trainDf["GrLivArea"] + trainDf["TotalBsmtSF"]
    # Variables that suffers from outliers
    trainDf['SalePrice'] = np.log1p(trainDf['SalePrice'])
    trainDf['LotFrontageLog'] = np.log1p(trainDf['LotFrontage'])
    trainDf['LotAreaLog'] = np.log1p(trainDf['LotArea'])
    trainDf['BsmtUnfSFLog'] = np.log1p(trainDf['BsmtUnfSF'])
    trainDf['AreaLog'] = np.log1p(trainDf['Area'])
    # trainDf['MasVnrAreaLog'] = np.log1p(trainDf['MasVnrArea'])
    # trainDf['BsmtFinSF1Log'] = np.log1p(trainDf['BsmtFinSF1'])
    # trainDf['BsmtFinSF2Log'] = np.log1p(trainDf['BsmtFinSF2'])
    # trainDf['TotalBsmtSFLog'] = np.log1p(trainDf['TotalBsmtSF'])
    # trainDf['1stFlrSFLog'] = np.log1p(trainDf['1stFlrSF'])
    # trainDf['GrLivAreaLog'] = np.log1p(trainDf['GrLivArea'])
    # trainDf['GarageAreaLog'] = np.log1p(trainDf['GarageArea'])
    # trainDf['WoodDeckSFLog'] = np.log1p(trainDf['WoodDeckSF'])
    # trainDf['OpenPorchSFLog'] = np.log1p(trainDf['OpenPorchSF'])
    # trainDf['EnclosedPorchLog'] = np.log1p(trainDf['EnclosedPorch'])
    # New features
    testDf['Area'] = testDf['LotArea'] * testDf['LotFrontage']
    testDf["TotalLA"] = testDf["GrLivArea"] + testDf["TotalBsmtSF"]
    # Variables that suffers from outliers
    testDf['SalePrice'] = np.log1p(testDf['SalePrice'])
    testDf['LotFrontageLog'] = np.log1p(testDf['LotFrontage'])
    testDf['LotAreaLog'] = np.log1p(testDf['LotArea'])
    testDf['BsmtUnfSFLog'] = np.log1p(testDf['BsmtUnfSF'])
    testDf['AreaLog'] = np.log1p(testDf['Area'])
    # testDf['MasVnrAreaLog'] = np.log1p(testDf['MasVnrArea'])
    # testDf['BsmtFinSF1Log'] = np.log1p(testDf['BsmtFinSF1'])
    # testDf['BsmtFinSF2Log'] = np.log1p(testDf['BsmtFinSF2'])
    # testDf['TotalBsmtSFLog'] = np.log1p(testDf['TotalBsmtSF'])
    # testDf['1stFlrSFLog'] = np.log1p(testDf['1stFlrSF'])
    # testDf['GrLivAreaLog'] = np.log1p(testDf['GrLivArea'])
    # testDf['GarageAreaLog'] = np.log1p(testDf['GarageArea'])
    # testDf['WoodDeckSFLog'] = np.log1p(testDf['WoodDeckSF'])
    # testDf['OpenPorchSFLog'] = np.log1p(testDf['OpenPorchSF'])
    # testDf['EnclosedPorchLog'] = np.log1p(testDf['EnclosedPorch'])
    return trainDf, testDf

# MinMaxScaling the variables
def minmaxScaling(trainDf, testDf, vars):
    scaler = MinMaxScaler()
    for var in vars:
        # fit on the training dataset
        scaler.fit(trainDf[[var]])
        # scale the training dataset
        trainDf[var+"MM"] = scaler.transform(trainDf[[var]])
        # scale the test dataset
        testDf[var+"MM"] = scaler.transform(testDf[[var]])
    return trainDf, testDf

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

# Fill missing values using prebuilt model
def fillNAUsingModel(i, model, varName):
    i[varName] = model.predict(i.drop(['SalePrice', varName]).values.reshape(1, -1))[0]
    return i

# Ordinal Encoding
def ordinalEncoding(df, vars, scaleMap):
    for var in vars:
        df[var] = df[var].replace(scaleMap)
    return df

# Calculate MAPE
def MAPE(actual, predicted):
    return (np.mean(np.abs((actual - predicted) / actual)) * 100).round(1)

# Encode dataframe
def oneHotEncoding(df, vars):
    for var in vars:
        # Join the encoded df
        df = df.join(pd.get_dummies(df[var], prefix=var))
    # Drop columns as they are now encoded
    return df.drop(vars, axis=1)

# CATBoost Model
def cbModel(trainDf, testDf, targetVar):
    cb = CatBoostRegressor(
        iterations=1000,
        depth=10,
        learning_rate=0.01,
        l2_leaf_reg= 0.1,
        loss_function='RMSE',
        eval_metric='MAE',
        random_strength=0.001,
        bootstrap_type='Bayesian',
        bagging_temperature=1,
        leaf_estimation_method='Newton',
        leaf_estimation_iterations=2,
        boosting_type='Ordered',
        feature_border_type='Median',
        random_seed=1234
    )
    # Fit data
    cb = cb.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return cb, cb.predict(testDf.drop(targetVar, 1))

# SVR Model
def svrModel(trainDf, testDf, targetVar):
    SVRModel = svm.SVR(kernel='rbf')
    # Fit data
    SVRModel = SVRModel.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return SVRModel, SVRModel.predict(testDf.drop(targetVar, 1))

# Random Forest Model
def rfModel(trainDf, testDf, targetVar):
    rf = RandomForestRegressor()
    # Fit data
    rf = rf.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return rf, rf.predict(testDf.drop(targetVar, 1))

# Gradient Boosting Model
def gbModel(trainDf, testDf, targetVar):
    # Tuning the parameters - Takes hours to find the best parameters
    # param_grid = {'n_estimators': range(20,101,20),
    #               'learning_rate': [0.01, 0.02, 0.05, 0.1],
    #               'max_depth': range(4,13,2),
    #               'min_samples_leaf': range(1,20, 3),
    #               'min_samples_split': range(2, 13, 2),
    #               'max_features': [0.25, 0.5, 0.75, 1.0],
    #               }
    # gsearch = GridSearchCV(estimator=GradientBoostingRegressor(),
    #                         param_grid=param_grid, n_jobs=4, cv=5, verbose=100)
    # gsearch.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    # print(gsearch.best_estimator_)
    gb = GradientBoostingRegressor(max_depth=4, max_features=0.75, min_samples_split=12, n_estimators=80)
    # Fit data
    gb = gb.fit(trainDf.drop(targetVar, 1), trainDf[targetVar])
    return gb, gb.predict(testDf.drop(targetVar, 1))
