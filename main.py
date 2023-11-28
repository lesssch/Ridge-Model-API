import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import io
import joblib
from model import lr_r

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    data = pd.read_json(item).to_dict()
    df = pd.DataFrame([data])
    predict = lr_r.predict(df)

    return predict


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> None:
    return ...