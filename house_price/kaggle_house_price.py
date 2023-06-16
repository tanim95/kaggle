# -*- coding: utf-8 -*-
"""kaggle_house_price.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OSdqTMhu2zRRJ7k0rWwCgwVBJRFxpDrB
"""

import pickle
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./house_price/kaggle_house_price.py')
df

df.isnull().sum().to_frame()

df.columns

sns.heatmap(df.isnull())

df.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# df['BsmtQual'].value_counts()
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].mean())
df['BsmtQual'] = df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtCond'] = df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtExposure'] = df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1'] = df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])
df['BsmtFinType2'] = df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['FireplaceQu'] = df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])
df['GarageType'] = df['GarageType'].fillna(df['GarageType'].mode()[0])

sns.heatmap(df.isnull())

columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'ExterCond', 'SaleType', 'SaleCondition', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
           'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

df_copy = df.copy()

test_df = pd.read_csv('/content/tst_modified.csv')
test_df

final_df = pd.concat([df_copy, test_df])


def create_dummy_variables(df, categorical_columns):
    modified_df = df.copy()

    for column in categorical_columns:
        dummy_columns = pd.get_dummies(
            modified_df[column], prefix=column, drop_first=True)
        modified_df = pd.concat(
            [modified_df.drop(column, axis=1), dummy_columns], axis=1)

    return modified_df


final_df = create_dummy_variables(final_df, columns)
# selecting non duplicate column
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

final_df.shape

correlation = final_df.corr()['SalePrice'].abs().sort_values(ascending=False)
correlation

top_features = correlation.index[:100]
top_features

new_dataset = final_df[top_features].copy()
new_dataset

train_data = new_dataset.loc[:1459, :]
test_data = new_dataset.loc[1459:, :].drop('SalePrice', axis=1)

X_train = train_data.drop('SalePrice', axis=1)
y_train = train_data['SalePrice']

model = XGBRegressor()

param = {
    'max_depth': [3, 5, 10, 15],
    'learning_rate': [0.2, 0.1, 0.01],
    'n_estimators': [100, 500, 900]
}

grid = GridSearchCV(estimator=model, param_grid=param,
                    cv=5, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

grid.best_params_, grid.best_score_

model = XGBRegressor(base_score=0.25, n_estimators=500,
                     learning_rate=0.01, max_depth=3, booster='gbtree')
model.fit(X_train, y_train)

pickle.dump(model, open('final_model.pkl', 'wb'))

y_pred = model.predict(test_data)

pred_dataframe = pd.DataFrame(y_pred)
sub_data = pd.read_csv('/content/sample_submission.csv')
dataset = pd.concat([sub_data['Id'], pred_dataframe], axis=1)

dataset.columns = ['Id', 'SalePrice']
dataset['Id'].fillna(0, inplace=True)
dataset['Id'] = dataset['Id'].astype('int32')
dataset = dataset.drop(dataset.index[-1])
dataset.to_csv('sample_submission_.csv', index=False)

dataset.shape
