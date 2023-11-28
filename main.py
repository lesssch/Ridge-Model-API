import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
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
    car = item.model_dump()
    car_instance = Item(**car)
    data = car_instance.model_dump()
    df = pd.DataFrame([data])
    df = df.drop_duplicates(keep="first", subset=df.columns.difference(["selling_price"]))
    df.reset_index(drop=True)

    df["mileage"] = df["mileage"].str.extract('(\d*\.?\d*)').astype(float)
    df["engine"] = df["engine"].str.extract('(\d+)').astype(float)
    df["max_power"] = df["max_power"].str.extract('(\d+\.?\d*)').astype(float)

    df = df.drop(["torque"], axis=1)

    df[["mileage", "engine", "max_power", "seats"]] = df[
        ["mileage", "engine", "max_power", "seats"]].fillna(
        df[["mileage", "engine", "max_power", "seats"]].median())

    df["engine"] = df["engine"].astype(int)
    df["seats"] = df["seats"].astype(int)

    df = df.drop(["selling_price", "name"], axis=1)
    df = pd.get_dummies(data=df, columns=["fuel", "seller_type", "transmission", "owner", "seats"],
                                 drop_first=True, prefix_sep="_", dtype=int)

    sc = StandardScaler()
    df[["year", "km_driven", "mileage", "engine", "max_power"]] = sc.fit_transform(
        df[["year", "km_driven", "mileage", "engine", "max_power"]])

    predict = lr_r.predict(df)

    return predict


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> None:
    return ...