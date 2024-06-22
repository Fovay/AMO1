import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path, index_col=0)
    scaler = StandardScaler()
    df['Temperature'] = scaler.fit_transform(df[['Temperature']])
    df.to_csv(output_path)
for filename in os.listdir('train'):
    preprocess_dataset(f'train/{filename}', f'train/{filename}')
for filename in os.listdir('test'):
    preprocess_dataset(f'test/{filename}', f'test/{filename}')
