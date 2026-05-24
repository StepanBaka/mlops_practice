# lab1/model_testing.py
import os
import sys
import pandas as pd
import pickle
from sklearn.metrics import r2_score

def create_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    df['temp_lag_1'] = df['temperature_scaled'].shift(1)
    return df.dropna()

def main():
    model_path = 'lab1/model.pkl'
    if not os.path.exists(model_path):
        print("Model file not found.", file=sys.stderr)
        sys.exit(1)
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    test_dir = 'lab1/test'
    test_files = [f for f in os.listdir(test_dir) if f.endswith('.csv')]
    
    X_test_list, y_test_list =[], []
    for file in test_files:
        df = pd.read_csv(os.path.join(test_dir, file))
        df_featured = create_autoregressive_features(df)
        X_test_list.append(df_featured[['temp_lag_1']])
        y_test_list.append(df_featured['temperature_scaled'])
        
    X_test = pd.concat(X_test_list, ignore_index=True)
    y_test = pd.concat(y_test_list, ignore_index=True)
    
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    
    # Форматированный вывод требуемой строки
    print(f"Model test accuracy is: {score:.3f}")

if __name__ == "__main__":
    main()