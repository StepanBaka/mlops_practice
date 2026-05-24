import pandas as pd


def apply_one_hot_encoding():
    file_path = "lab4/data/titanic_features.csv"

    df = pd.read_csv(file_path)

    df_encoded = pd.get_dummies(df, columns=["Sex"], drop_first=False)

    for column in df_encoded.columns:
        if df_encoded[column].dtype == bool:
            df_encoded[column] = df_encoded[column].astype(int)

    df_encoded.to_csv(file_path, index=False)

    print("Step 3: One-Hot Encoding applied to Sex feature.")
    print(f"Saved to: {file_path}")
    print(f"Columns: {list(df_encoded.columns)}")


if __name__ == "__main__":
    apply_one_hot_encoding()