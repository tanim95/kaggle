
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./gemstone/train.csv')

df_test = pd.read_csv('./gemstone/test.csv')
df_test.head()

df.info()
df.isnull().sum()
df.duplicated()
df.corr()['price'].abs().sort_values(ascending=False)

df_copy = df.copy()
new_df = df_copy.drop(['id', 'depth'], axis=1)
print(new_df)
# dropping the last  column as it has more than 6 NaN value
new_df = df.drop(df.index[-1])


converted_df = new_df.drop('id', axis=1)
converted_df = pd.get_dummies(
    converted_df, columns=['cut', 'color', 'clarity'], drop_first=True)
converted_df.drop('depth', axis=1, inplace=True)
print(converted_df)

df.shape

# Dealing with outliers
plt.figure(figsize=(10, 8), dpi=100)
sns.boxplot(converted_df, x='price')

converted_df.describe()['price']

final_df = converted_df.drop(converted_df[converted_df['price'] > 11500].index)
final_df

columns = final_df.columns

plt.figure(figsize=(10, 8), dpi=100)
sns.boxplot(final_df, x='price')

scaler = StandardScaler()
final_df = scaler.fit_transform(final_df)
final_df

final_df = pd.DataFrame(final_df, columns=columns)
final_df

X_train = final_df.drop(['price'], axis=1)
y_train = final_df['price']

# test data preprocessing
new_test_df = df_test.drop(['id', 'depth'], axis=1)
new_test_df

new_test_df.isnull().sum()

new_test_df.duplicated()

converted_test_df = pd.get_dummies(
    new_test_df, columns=['cut', 'color', 'clarity'], drop_first=True)
converted_test_df

test_columns = converted_test_df.columns

scaler_test = StandardScaler()
final_test_df = scaler_test.fit_transform(converted_test_df)
final_test_df

final_test_df = pd.DataFrame(final_test_df, columns=test_columns)
final_test_df

X_train = X_train.iloc[:129050, :]
y_train = y_train.iloc[:129050]


rfr_model = RandomForestRegressor()

param = {
    'n_estimators': [100, 200, 500],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_model = GridSearchCV(rfr_model, param_grid=param)

grid_model.fit(X_train, y_train)

grid_model.best_estimator_, grid_model.best_params_

y_pred = rfr_model.predict(final_test_df)

pred_dataframe = pd.DataFrame(y_pred)
sub_data = pd.read_csv('/content/sample_submission.csv')
dataset = pd.concat([sub_data['id'], pred_dataframe], axis=1)

dataset.columns = ['id', 'price']
# dataset['Id'].fillna(0, inplace=True)
# dataset['Id'] = dataset['Id'].astype('int32')
# dataset = dataset.drop(dataset.index[-1])
dataset.to_csv('sample_submission_.csv', index=False)

"""#XGBRegressor model"""

xgb_model = XGBRegressor()

param_2 = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.001],
    'reg_alpha': [0, 0.1, 0.5],
    'gamma': [0, 0.1, 0.5]
}

xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=param_2)

# xgb_grid.fit(X_train,y_train)

# y_pred_2 =  xgb_grid(X_test)
