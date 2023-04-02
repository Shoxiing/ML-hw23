
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')


from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import json
import pickle
from sklearn.preprocessing import StandardScaler
from starlette.responses import FileResponse

app = FastAPI()
scaler = StandardScaler()
loaded_model = pickle.load(open('auto_model.sav','rb'))


class Item(BaseModel):
    name: str
    year: int
    selling_price: str
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


def prep_data(df_test):
    df_test["engine"] = df_test["engine"].str.replace(r"[^\d\.]", "", regex=True)
    df_test["mileage"] = df_test["mileage"].str.replace(r"[^\d\.]", "", regex=True)
    df_test["max_power"] = df_test["max_power"].str.replace(r"[^\d\.]", "", regex=True)
    df_test["engine"] = df_test['engine'].astype(float)
    df_test['max_power'] = pd.to_numeric(df_test['max_power'])
    df_test["mileage"] = df_test["mileage"].astype(float)
    df_test.drop('torque', axis=1, inplace=True)
    df_test = df_test.fillna(df_test.median())
    df_test = df_test.drop(['name', 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price'], axis=1)
    return df_test



@app.post("/predict_item")
def predict_item(item: Item) -> float:
    input_data = item.json()
    input_dictionary = json.loads(input_data)
    df = pd.DataFrame.from_records(input_dictionary, index=[0])
    df = prep_data(df)
    df = scaler.fit_transform(df)
    price = loaded_model.predict(df)
    return float('{:.3f}'.format(float(price)))

@app.post("/predict_items")
def upload_csv(csv_file: UploadFile):
    df1 = pd.read_csv(csv_file.file)
    df_pr= prep_data(df1)
    df_sc = scaler.fit_transform(df_pr)
    predict_price = loaded_model.predict(df_sc)
    df1['predict_price'] = pd.DataFrame(predict_price)
    df1.to_csv('car_price_0.csv')
    return FileResponse(path='car_price_0.csv', media_type='text/csv',filename='car_price_0.csv')








