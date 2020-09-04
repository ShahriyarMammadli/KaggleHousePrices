# KaggleHousePrices
This solution had **0.12138** Root-Mean-Squared-Error (RMSE) which is in top **10%** in the leaderboard.

## Requirements
The list of the libraries that needed to be installed are:  
* **pandas**
* **scikit-learn**
* **xgboost**
* **catboost (optional, because it is not selected as a best performing algorithm)**
* **matplotlib**

### Exploratory Data Analysis(EDA) and Feature Engineering
**Note:** To avoid data leakage parameters in feature engineering fitted on training set.

**Step 1:** Missing value analysis
Missing value analysis have been made to see the variables that have missing values and what kind of treatment should be applied to that variable.
For training set:  
| Variable | Missing samples (Percentage) |
| :-----: | :-: |
| Electrical | 0.1% |
| MasVnrType | 0.5% |
| MasVnrArea | 0.5% |
| BsmtQual | 2.5% |
| BsmtCond | 2.5% |
| BsmtFinType1 | 2.5% |
| BsmtExposure | 2.6% |
| BsmtFinType2 | 2.6% |
| GarageCond | 5.5% |
| GarageQual | 5.5% |
| GarageFinish | 5.5% |
| GarageType | 5.5% |
| GarageYrBlt | 5.5% |
| LotFrontage | 17.7% |
| FireplaceQu | 47.3% |
| Fence | 80.8% |
| Alley | 93.8% |
| MiscFeature | 96.3% |
| PoolQC | 99.5% |

For testing set:  

| Variable | Missing samples (Percentage) |
| :-----: | :-: |
| TotalBsmtSF | 0.1% |
| GarageArea | 0.1% |
| GarageCars | 0.1% |
| KitchenQual | 0.1% |
| BsmtUnfSF | 0.1% |
| BsmtFinSF2 | 0.1% |
| BsmtFinSF1 | 0.1% |
| SaleType | 0.1% |
| Exterior1st | 0.1% |
| Exterior2nd | 0.1% |
| Functional | 0.1% |
| Utilities | 0.1% |
| BsmtHalfBath | 0.1% |
| BsmtFullBath | 0.1% |
| MSZoning | 0.3% |
| MasVnrArea | 1.0% |
| MasVnrType | 1.1% |
| BsmtFinType2 | 2.9% |
| BsmtFinType1 | 2.9% |
| BsmtQual | 3.0% |
| BsmtExposure | 3.0% |
| BsmtCond | 3.1% |
| GarageType | 5.2% |
| GarageFinish | 5.3% |
| GarageQual | 5.3% |
| GarageCond | 5.3% |
| GarageYrBlt | 5.3% |
| LotFrontage | 15.6% |
| FireplaceQu | 50.0% |
| Fence | 80.1% |
| Alley | 92.7% |
| MiscFeature | 96.5% |
| PoolQC | 99.8% |

**Step 2:** Treating missing values
Following variables' missing values are filled with 'None' since as mentioned in the explanation of these variables NA means house does not have this 'feature'. e.g.
> NA 	No alley access

_PoolQC_, _MiscFeature_, _Alley_, _Fence_, _FireplaceQu_, _GarageType_, _GarageFinish_, _GarageQual_, _GarageCond_, _BsmtFinType2_, _BsmtExposure_, _BsmtFinType1_, _BsmtCond_, _BsmtQual_, _GarageYrBlt_

GarageYrBlt indicates the year when the garage is built but properties that does not have garage is empty and since year is categorical variable I filled the NAs of these properties with 'None'.

_MasVnrType_ has only couple of missing values. In description of this variable, it is mentioned that if sample has not *Masonry veneer type* then it is labeled as 'None'. Thus, I filled the missing values with 'Other'.

_MasVnrArea_ has missing values where _MasVnrType_ of that sample is 'None' and 'Other' thus I replaced these missing values with 0. 

_Electrical_ has few missing values I replaced them with most freqeunt value in the variable.

Following variables also have few missing values and they replaced with 0. 
_TotalBsmtSF_, _GarageArea_, _GarageCars_, _BsmtUnfSF_, _BsmtFinSF2_, _BsmtFinSF1_, _BsmtHalfBath_, _BsmtFullBath_.

TotalBsmtSF is filled with 0 because, a house has also NA for other basement features which means it does not have a basement and houses with no basement has TotalBsmtSF of 0, thus this sample is also filled with 0.

Following variables are also filled wiht most frequent value in that variable.
_KitchenQual_, _SaleType_, _Exterior1st_, _Exterior2nd_, _Functional_, _Utilities_, _MSZoning_.

_LotFrontage_ variable has many missing values which needs special treatment. For this purpose, I build a Random Forest Regressor by using training data to predict it.

New variables are created where _Area_ is sum of _LotArea_ and _LotFrontage_ and _TotalLA_ is sum of _GrLivArea_ and _TotalBsmtSF_.

**Step 3:** Outlier handling
I plotted the scatter plot of the variables vs dependent variable to see the outliers. In this way, I removed the outliers from the variables. However, this technique did not help a lot, an it needs more careful analysis and treatment. Due to time constraints, I did log transformation to these variables to handle the outliers. _LotAreaLog_, _LotFrontageLog_, _BsmtUnfSFLog_, _AreaLog_ variableas are cretaed fur this purpose. 

**Step 4:** Feature Scaling
Following variables are scaled using mix-max scaling: _BsmtFinSF2_, _LowQualFinSF_, _WoodDeckSF_, _OpenPorchSF_, _EnclosedPorch_, _3SsnPorch_, _ScreenPorch_, _MiscVal_.

**Step 5:** Variables are encoded depending on their type (ordinal, categorical or numerical).

## Modelling
Random Forest, XGBoost, Gradien Boosting, SVM, CatBoost are used.

## Result
| Model | Score  |
| :-----: | :-: |
| SVM | 0.76555 |
| RF | 0.74401 |
| GB | 0.79186 |
| Ensemble | 0.77272 |

As a result, GB performed best with parameters of _n_estimators=100, max_features='sqrt'_. The why of the ensemble model performed worse than GB is due to SVM and RF were mostly failed together to predict for Survived for a sample. 

# ToDo
In total 7-8 hours are spent on building this model, thus, it can be improved by putting additional effort. While analyzing the data, it was obvious that all data samples and features have some information, therefore, all of them must be somehow used to obtain maximum accuracy. Some of the possible improvements are:
* Predicting the missing samples of Embarked variable
* Tuning the hyperparameters
* Eliminating the '.' and '/' characters from the ticket samples (I noticed it after submitting the result)
* Using other algorithms especially I am curious about CatBoost's performance.

