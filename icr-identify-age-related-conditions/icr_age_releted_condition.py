# -*- coding: utf-8 -*-
"""icr_age_releted_condition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xZnwqjtz_NWXEvcnOrhK_1jq9E_4hZef
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/train.csv')
df

df.isnull().sum()

df_copy = df.copy()

df_copy['EJ'].unique()

df_copy = df_copy.dropna(subset=['BQ', 'EL'])

dummy = pd.get_dummies(df_copy['EJ'], prefix='EJ', drop_first=True)
df_copy = pd.concat([df_copy.drop('EJ', axis=True), dummy], axis=1)

sns.heatmap(df_copy.isnull())

greek = pd.read_csv('/content/greeks.csv')
greek

greek['Epsilon'].value_counts()['Unknown']

greek.drop('Epsilon', axis=1, inplace=True)

greek.columns

columns = ['Alpha', 'Beta', 'Gamma', 'Delta']

new_greek = greek.copy()
for i in columns:
    dummy = pd.get_dummies(greek[i], prefix=i, drop_first=True)
    new_greek = pd.concat([new_greek.drop(i, axis=1), dummy], axis=1)

new_greek

merged_df = pd.merge(df_copy, new_greek, on='Id')
merged_df

merged_df.isnull().sum()

final_df = merged_df.drop('Id', axis=1)
final_df
