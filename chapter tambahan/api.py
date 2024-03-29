""" 
API for the Tip Prediction Model
"""
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Bikin class untuk Model
class TipPredictionModel:
    """Model Tip Prediction Class"""

    def __init__(self, transformer_filename:str, model_filename:str) -> None:
        with open(transformer_filename, 'rb') as f:
            self.transformer = pickle.load(f)
        with open(model_filename, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, data:pd.DataFrame) -> float:
        data = self.transformer.transform(data)
        prediction = self.model.predict(data)
        return prediction[0]

# Bikin class untuk API Request
class TipPredictionRequest(BaseModel):
    total_bill: float
    sex: str
    smoker: str
    day: str
    time: str
    size: int

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'total_bill': [self.total_bill],
                'sex': [self.sex],
                'smoker': [self.smoker],
                'day': [self.day],
                'time': [self.time],
                'size': [self.size]
            }
        )

# Bikin API dengan FastAPI
app = FastAPI()
model = TipPredictionModel(transformer_filename='tip_preprocessor.pkl', model_filename='tip_model.pkl')

# Routing Informasi Dasar
@app.get('/')
def index():
    return {'message': 'Hello, this is Tip Prediction Model API'}

# Routing Prediksi
@app.post('/predict')
def predict(request: TipPredictionRequest):
    data = request.to_df()
    prediction = model.predict(data)
    return {'prediction': prediction}

# Jalanin API
if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)

http://localhost:8000/docs