import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


np.random.seed(101)


@pytest.fixture(scope="module")
def clean_dataset():
    """
    Чистые данные с линейной зависимостью:
    y = 3x + 5 + небольшой шум
    """
    x = np.linspace(0, 20, 150).reshape(-1, 1)
    y = 3.0 * x.squeeze() + 5.0 + np.random.normal(0, 1.0, 150)

    return pd.DataFrame(
        {
            "feature_x": x.squeeze(),
            "target_y": y,
        }
    )


@pytest.fixture(scope="module")
def noised_dataset(clean_dataset):
    """
    Зашумлённые данные с выбросами.
    На них качество модели должно ухудшиться.
    """
    df = clean_dataset.copy()
    n_samples = len(df)

    df["target_y"] += np.random.normal(0, 15.0, n_samples)

    outlier_indices = np.random.choice(n_samples, 10, replace=False)
    df.loc[outlier_indices, "target_y"] *= -5.0

    return df


@pytest.fixture(scope="module")
def production_model(clean_dataset):
    """
    Модель, обученная на чистых данных.
    """
    x_train = clean_dataset[["feature_x"]]
    y_train = clean_dataset["target_y"]

    model = LinearRegression()
    model.fit(x_train, y_train)

    return model


def test_model_accuracy_on_clean_data(production_model, clean_dataset):
    """
    Проверяем, что на чистых данных модель показывает высокое качество.
    """
    x = clean_dataset[["feature_x"]]
    y_true = clean_dataset["target_y"]

    y_pred = production_model.predict(x)
    score = r2_score(y_true, y_pred)

    print(f"R2 score on clean data: {score:.4f}")

    assert score >= 0.95, (
        f"Model underperformed on clean data. "
        f"R2 = {score:.4f}"
    )


def test_model_robustness_on_noised_data(production_model, noised_dataset):
    """
    Проверяем качество на зашумлённых данных.
    Этот тест специально может падать, чтобы показать деградацию данных.
    """
    x = noised_dataset[["feature_x"]]
    y_true = noised_dataset["target_y"]

    y_pred = production_model.predict(x)
    score = r2_score(y_true, y_pred)

    print(f"R2 score on noised data: {score:.4f}")

    assert score >= 0.70, (
        f"DATA DEGRADATION DETECTED: "
        f"Model accuracy dropped to R2 = {score:.4f}. "
        f"Deploy halted."
    )