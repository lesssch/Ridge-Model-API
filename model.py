import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib


df_train = pd.read_csv('https://raw.githubusercontent.com/hse-mlds/ml/main/hometasks/HT1/cars_train.csv')

df_train = df_train.drop_duplicates(keep="first", subset=df_train.columns.difference(["selling_price"]))
df_train.reset_index(drop=True)

df_train["mileage"] = df_train["mileage"].str.extract('(\d*\.?\d*)').astype(float)
df_train["engine"] = df_train["engine"].str.extract('(\d+)').astype(float)
df_train["max_power"] = df_train["max_power"].str.extract('(\d+\.?\d*)').astype(float)

df_train = df_train.drop(["torque"], axis=1)

df_train[["mileage", "engine", "max_power", "seats"]] = df_train[["mileage", "engine", "max_power", "seats"]].fillna(df_train[["mileage", "engine", "max_power", "seats"]].median())

df_train["engine"] = df_train["engine"].astype(int)
df_train["seats"] = df_train["seats"].astype(int)

y_train = df_train["selling_price"]

X_train_cat = df_train.drop(["selling_price", "name"], axis=1)
X_train_cat = pd.get_dummies(data=X_train_cat, columns=["fuel", "seller_type", "transmission", "owner", "seats"], drop_first=True, prefix_sep="_", dtype=int)

sc = StandardScaler()
X_train_cat = sc.fit_transform(X_train_cat)

lr_r = Ridge(alpha=5)
lr_r.fit(X_train_cat, y_train)

joblib.dump(lr_r, "ridge_model.joblib")
