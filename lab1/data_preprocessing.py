# lab1/data_preprocessing.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def main():
    scaler = StandardScaler()
    train_dir = 'lab1/train'
    test_dir = 'lab1/test'
    
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    
    
    train_dfs = [pd.read_csv(os.path.join(train_dir, f)) for f in train_files]
    full_train_df = pd.concat(train_dfs, ignore_index=True)
    
    
    scaler.fit(full_train_df[['temperature']])
    logging.info(f"Scaler fitted. Mean: {scaler.mean_}, Var: {scaler.var_}")
    
    
    for file, df in zip(train_files, train_dfs):
        df['temperature_scaled'] = scaler.transform(df[['temperature']])
        df.to_csv(os.path.join(train_dir, file), index=False)
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    for file in test_files:
        df = pd.read_csv(os.path.join(test_dir, file))
        df['temperature_scaled'] = scaler.transform(df[['temperature']])
        df.to_csv(os.path.join(test_dir, file), index=False)
    with open('lab1/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logging.info("Data preprocessing completed and scaler saved.")

if __name__ == "__main__":
    main()