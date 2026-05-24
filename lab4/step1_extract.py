import os

from catboost.datasets import titanic


def extract_base_dataset():
    train_df, _ = titanic()

    selected_columns = ["Pclass", "Sex", "Age"]
    subset = train_df[selected_columns]

    os.makedirs("lab4/data", exist_ok=True)

    subset.to_csv("lab4/data/titanic_features.csv", index=False)

    print("Step 1: Base dataset extracted.")
    print(f"Saved to: lab4/data/titanic_features.csv")
    print(f"Shape: {subset.shape}")


if __name__ == "__main__":
    extract_base_dataset()