import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

X_train, y_train = [], []
for filename in os.listdir('train'):
    df = pd.read_csv(f'train/{filename}', index_col=0)
    X_train.append(pd.to_datetime(df.index).map(pd.Timestamp.toordinal).values.reshape(-1, 1))
    y_train.append(df['Temperature'].values.reshape(-1, 1))
X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')
