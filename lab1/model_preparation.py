# lab1/model_preparation.py
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import logging

logging.basicConfig(level=logging.INFO)

def create_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация лаговых признаков для моделирования временных рядов."""
    df['temp_lag_1'] = df['temperature_scaled'].shift(1)
    # Удаление строк с NaN, образовавшихся в результате сдвига
    return df.dropna()

def main():
    train_dir = 'lab1/train'
    train_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
    
    X_train_list, y_train_list =[], []
    for file in train_files:
        df = pd.read_csv(os.path.join(train_dir, file))
        df_featured = create_autoregressive_features(df)
        X_train_list.append(df_featured[['temp_lag_1']])
        y_train_list.append(df_featured['temperature_scaled'])
        
    X_train = pd.concat(X_train_list, ignore_index=True)
    y_train = pd.concat(y_train_list, ignore_index=True)
    
    # Инициализация и подгонка модели
    model = LinearRegression()
    model.fit(X_train, y_train)
    logging.info(f"Model trained. Coefficients: {model.coef_}, Intercept: {model.intercept_}")
    
    with open('lab1/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model serialized to lab1/model.pkl")

if __name__ == "__main__":
    main()