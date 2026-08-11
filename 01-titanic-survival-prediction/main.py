from enum import Enum
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import custom_transformers  # noqa: F401


class SexEnum(str, Enum):
    male = "male"
    female = "female"



app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A FastAPI web service that takes passenger data and applies a Machine Learning model to predict survival probability.",
    version="1.1.0",
)


MODEL_PATH = "model/titanic_pipeline.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")


class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger Class (1, 2, or 3)")
    Name: str = Field(..., description="Full Name of the Passenger")
    Sex: SexEnum = Field(..., description="Gender ('male' or 'female')")
    Age: Optional[float] = Field(None, ge=0.0, le=120.0, description="Age in years (optional)")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Ticket: str = Field(..., description="Ticket number/string")
    Fare: float = Field(..., ge=0.0, description="Ticket fare paid")
    Cabin: Optional[str] = Field(None, description="Cabin number (Optional/Missing for many)")
    Embarked: str = Field(None, description="Port of Embarkation ('S', 'C', or 'Q')")

    model_config = {
        "use_enum_values": True,
        "json_schema_extra": {
            "examples": [
                {
                    "Pclass": 3,
                    "Name": "Braund, Mr. Owen Harris",
                    "Sex": "male",
                    "Age": 22.0,
                    "SibSp": 1,
                    "Parch": 0,
                    "Ticket": "A/5 21171",
                    "Fare": 7.25,
                    "Cabin": None,
                    "Embarked": "S",
                }
            ]
        },
    }


@app.get("/")
def health_check():
    """Health check endpoint to confirm API status."""
    return {
        "status": "healthy",
        "service": "Titanic Survival Prediction API",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(passenger: PassengerData):
    """
    Accepts complete passenger dataset row, converts data into DataFrame,
    and returns survival prediction and probabilities.
    """
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model file is missing or failed to load on backend startup.",
        )

    data_dict = passenger.model_dump()
    input_df = pd.DataFrame([data_dict])

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        survival_probability = float(probabilities[1])

        return {
            "survived": prediction,
            "survival_probability": round(survival_probability, 4),
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Inference execution error: {str(e)}",
        )