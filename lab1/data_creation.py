# lab1/data_creation.py
import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_temperature_data(n_samples: int = 1000, anomaly_fraction: float = 0.05) -> pd.DataFrame:
    """
    Генерация синтетических данных дневной температуры с гармоническим трендом, 
    стохастическим шумом и случайными аномалиями.
    """
    time = np.linspace(0, 4 * np.pi, n_samples)
    temperature = 20.0 + 10.0 * np.sin(time) + np.random.normal(0, 0.5, n_samples)
    
    n_anomalies = int(n_samples * anomaly_fraction)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    temperature[anomaly_indices] += np.random.normal(3, 1, n_anomalies)
    
    return pd.DataFrame({'day_index': range(n_samples), 'temperature': temperature})

def main():
    os.makedirs('lab1/train', exist_ok=True)
    os.makedirs('lab1/test', exist_ok=True)
    
    np.random.seed(42)
    
    for i in range(3):
        data = generate_temperature_data(n_samples=1500)
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        train_file = f'lab1/train/dataset_{i}.csv'
        test_file = f'lab1/test/dataset_{i}.csv'
        
        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)
        logging.info(f"Dataset {i} created. Train size: {len(train_data)}, Test size: {len(test_data)}")

if __name__ == "__main__":
    main()