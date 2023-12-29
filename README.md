# Budget-tricks-by-QTR
import pandas as pd
import numpy as np
import os

# !pip install scikit-learn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

df = pd.read_csv('/Users/tgalarneau2024/2023-12-28T12_51_02.793Z-transactions.csv')

df['Date'] = pd.to_datetime(df['Date'])  # Replace 'date_column' with your date column name

df['quarter'] = df['Date'].dt.to_period('Q')

quarterly_spending = df.groupby('quarter')['Amount'].sum()  # Replace 'spending_column' with your relevant column

quarterly_spending = quarterly_spending.reset_index()

df.head()

# df = df.drop(['Tax Deductible', 'Ignored From', 'Note', 'Custom Name',], axis=1)

df.info()


quarterly_spending['quarter_num'] = quarterly_spending['quarter'].dt.qyear * 4 + quarterly_spending['quarter'].dt.quarter

X = quarterly_spending[['quarter_num']]  # Feature
y = quarterly_spending['Amount']  # Target

# Run linear regression
model = LinearRegression()
model.fit(X, y)


# df['Category'].value_counts().plot(kind= 'pie', autopct='%1.1f%%', cmap= 'tab20c', figsize=(20,12))
