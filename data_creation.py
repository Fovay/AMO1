import os
import numpy as np
import pandas as pd


os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)
def create_dataset(filename):
    dates = pd.date_range('20230101', periods=100)
    data = np.random.randn(100).cumsum()  # Random walk data
    df = pd.DataFrame(data, index=dates, columns=['Temperature'])
    df.to_csv(filename)
for i in range(5):
    create_dataset(f'train/data_{i}.csv')

for i in range(2):
    create_dataset(f'test/data_{i}.csv')
