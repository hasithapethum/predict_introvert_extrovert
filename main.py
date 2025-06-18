from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import pandas as pd  # <-- Add this import

app = FastAPI()

# Load model
model = joblib.load("personality_model.pkl")

# Define input schema
class InputData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: float
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: float
    Friends_circle_size: float
    Post_frequency: float

@app.post("/predict")
def predict(data: InputData):
    features = [
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_event_attendance,
        data.Going_outside,
        data.Drained_after_socializing,
        data.Friends_circle_size,
        data.Post_frequency
    ]

    # Create DataFrame with correct column names
    columns = [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
    ]
    input_df = pd.DataFrame([features], columns=columns)
    prediction = model.predict(input_df)
    
    if int(prediction[0]) == 0:
        return {"prediction": "Introvert"}
    else:
        return {"prediction": "Extrovert"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)