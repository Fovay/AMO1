import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

model = joblib.load('model.pkl')
for filename in os.listdir('test'):
    df = pd.read_csv(f'test/{filename}', index_col=0)
    X_test = pd.to_datetime(df.index).map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    y_test = df['Temperature'].values.reshape(-1, 1)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE for {filename}: {mse}')
