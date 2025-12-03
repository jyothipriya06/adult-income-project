from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Create FastAPI app
app = FastAPI(
    title="Adult Income Prediction API",
    description="Predicts whether a person earns >50K based on census features.",
    version="1.0.0"
)

# 1. Load the trained model (Pipeline with preprocessing + RandomForest)
#    Adjust the path if your model is in a different location.
MODEL_PATH = "C:/Users/priya/OneDrive/Desktop/adult-income-project/models/final_model.joblib"
model = joblib.load(MODEL_PATH)

# 2. Define input schema (what the API expects in the request body)
#    These names are Python-friendly; we will map them to the original column names.
class IncomeFeatures(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int      # maps to "education-num"
    marital_status: str     # maps to "marital-status"
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int       # maps to "capital-gain"
    capital_loss: int       # maps to "capital-loss"
    hours_per_week: int     # maps to "hours-per-week"
    native_country: str     # maps to "native-country"


# 3. Root endpoint - simple health check
@app.get("/")
def read_root():
    return {"message": "Adult Income Prediction API is running"}


# 4. Prediction endpoint
@app.post("/predict")
def predict_income(features: IncomeFeatures):
    """
    Accepts user features and returns:
    - predicted class (0 = <=50K, 1 = >50K)
    - probability of earning >50K
    """

    # Map Pydantic model → dict with original training column names
    data = {
        "age": features.age,
        "workclass": features.workclass,
        "fnlwgt": features.fnlwgt,
        "education": features.education,
        "education-num": features.education_num,
        "marital-status": features.marital_status,
        "occupation": features.occupation,
        "relationship": features.relationship,
        "race": features.race,
        "sex": features.sex,
        "capital-gain": features.capital_gain,
        "capital-loss": features.capital_loss,
        "hours-per-week": features.hours_per_week,
        "native-country": features.native_country,
    }

    # Convert to DataFrame with a single row
    input_df = pd.DataFrame([data])

    # Get predictions from the pipeline
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]  # probability of class 1 (>50K)

    return {
        "predicted_income_class": int(pred_class),
        "probability_gt_50k": float(pred_proba)
    }
