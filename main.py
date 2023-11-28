import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
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

class Schema(BaseModel):
    year: float
    km_driven: float
    mileage: float
    engine: float
    max_power: float
    fuel_Diesel: int = 0
    fuel_LPG: int = 0
    fuel_Petrol: int = 0
    seller_type_Individual: int = 0
    seller_type_Trustmark_Dealer: int = 0
    transmission_Manual: int = 0
    owner_Fourth_Above_Owner: int = 0
    owner_Second_Owner: int = 0
    owner_Test_Drive_Car: int = 0
    owner_Third_Owner: int = 0
    seats_4: int = 0
    seats_5: int = 0
    seats_6: int = 0
    seats_7: int = 0
    seats_8: int = 0
    seats_9: int = 0
    seats_10: int = 0
    seats_14: int = 0



@app.post("/predict_item")
def predict_item(item: Item) -> dict:
    car = item.model_dump()
    car_instance = Item(**car)
    data = car_instance.model_dump()
    df = pd.DataFrame([data])

    df["mileage"] = df["mileage"].str.extract('(\d*\.?\d*)').astype(float)
    df["engine"] = df["engine"].str.extract('(\d+)').astype(float)
    df["max_power"] = df["max_power"].str.extract('(\d+\.?\d*)').astype(float)

    df = df.drop(["torque"], axis=1)
    data = df.to_dict()

    # df["engine"] = df["engine"].astype(int)
    # df["seats"] = df["seats"].astype(int)
    #
    # df.replace(" ", "_", regex=True, inplace=True)
    # df.replace("&_", "", regex=True, inplace=True)
    #
    # df = df.drop(["selling_price", "name"], axis=1)
    # df = pd.get_dummies(data=df, columns=["fuel", "seller_type", "transmission", "owner", "seats"], prefix_sep="_", dtype=int)
    #
    # sc = StandardScaler()
    # df[["year", "km_driven", "mileage", "engine", "max_power"]] = sc.fit_transform(
    #     df[["year", "km_driven", "mileage", "engine", "max_power"]])
    #
    # df.to_dict()
    # car = Schema(**df)
    # data = car.model_dump()
    # df = pd.DataFrame([data])
    #
    # predict = lr_r.predict(df)

    return data


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> None:
    return ...