import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path(__file__).parent / "models" / "catboost_model.pkl"

app = FastAPI(
    title="MLOps Titanic Inference Service",
    description="API для предсказания выживания пассажира Titanic",
    version="1.0.0",
)


try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None


class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., description="Passenger class: 1, 2 or 3")
    Name: str = Field(default="Unknown", description="Passenger name")
    Sex: str = Field(..., description="Passenger sex: male/female")
    Age: float = Field(default=0.0, description="Passenger age")
    SibSp: int = Field(default=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(default=0, description="Number of parents/children aboard")
    Ticket: str = Field(default="Unknown", description="Ticket number")
    Fare: float = Field(default=0.0, description="Ticket fare")
    Cabin: str = Field(default="Unknown", description="Cabin number")
    Embarked: str = Field(default="Unknown", description="Port of embarkation")


@app.get("/")
def root():
    return {
        "message": "Titanic ML inference service is running",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact not found")

    return {
        "status": "healthy",
        "model": "CatBoostClassifier",
    }


@app.post("/predict")
def predict_survival(passenger: PassengerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    input_data = passenger.model_dump()

    df = pd.DataFrame([input_data])

    try:
        prediction = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]

        survival_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "survival_probability": round(survival_probability, 4),
            "interpretation": "Survived" if prediction == 1 else "Deceased",
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(error)}",
        )