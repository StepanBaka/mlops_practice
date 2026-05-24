import os
import sys
import pickle
import logging

import pandas as pd
from catboost.datasets import titanic
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def fetch_and_prepare_data():
    os.makedirs("lab2/data", exist_ok=True)

    train_df, _ = titanic()

    logging.info(f"Dataset loaded. Shape: {train_df.shape}")

    train_df = train_df.dropna(subset=["Survived"])

    y = train_df["Survived"]
    X = train_df.drop("Survived", axis=1)

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in X.select_dtypes(include=["float64", "int64"]).columns:
        X[col] = X[col].fillna(X[col].mean())

    X.to_csv("lab2/data/X_processed.csv", index=False)
    y.to_csv("lab2/data/y_processed.csv", index=False)

    with open("lab2/data/cat_features.pkl", "wb") as f:
        pickle.dump(cat_features, f)

    logging.info("Data preparation completed successfully.")
    logging.info(f"Categorical features: {cat_features}")


def train_and_evaluate_model():
    os.makedirs("lab2/models", exist_ok=True)

    X = pd.read_csv("lab2/data/X_processed.csv")
    y = pd.read_csv("lab2/data/y_processed.csv").squeeze()

    with open("lab2/data/cat_features.pkl", "rb") as f:
        cat_features = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    train_pool = Pool(
        data=X_train,
        label=y_train,
        cat_features=cat_features,
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        cat_features=cat_features,
    )

    model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=20,
    )

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    logging.info(f"Model test accuracy is: {accuracy:.4f}")
    logging.info("\n" + classification_report(y_test, predictions))

    with open("lab2/models/catboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    logging.info("Model saved to lab2/models/catboost_model.pkl")


def main():
    if len(sys.argv) < 2:
        print("Usage: python lab2/data_pipeline.py [fetch|train]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "fetch":
        fetch_and_prepare_data()
    elif command == "train":
        train_and_evaluate_model()
    else:
        print("Unknown command. Use: fetch or train")
        sys.exit(1)


if __name__ == "__main__":
    main()