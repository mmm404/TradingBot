import os.path

import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib
import time


def create_file(li):
    if not os.path.exists(r'.\prediction.csv','a'):
        with open(r'.\prediction.csv','w') as data_file:
            data_file.write(sum(li)/4)
    else:
        with open(r'.\prediction.csv','a') as data_file:
            data_file.write(sum(li)/4)

def make_predictions(model_path, input_row):
    model = load_model(model_path)
    sequence_length = input_row.shape[1]
    num_features = input_row.shape[2]
    input_data = input_row.reshape(1, sequence_length, num_features)

    prediction = model.predict(input_data)[0]

    return prediction.tolist()


def get_api_data():
    api_key = 'ZASMYPGR0SQTETJN'
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "EUR",
        "to_currency": "USD",
        "apikey": api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    currency_data = data.get('Realtime Currency Exchange Rate', {})
    exchange_rate = float(currency_data.get('5. Exchange Rate', 0))
    bid_price = float(currency_data.get('8. Bid Price', 0))
    ask_price = float(currency_data.get('9. Ask Price', 0))

    if exchange_rate == 0 and bid_price == 0 and ask_price == 0:
        print("Error: 'Realtime Currency Exchange Rate' not found in API response.")
        return None
    duplicated_value = ask_price
    features = [exchange_rate, bid_price, ask_price, duplicated_value]
    features = [float(value) for value in features]

    return [features]

def reshape_live_data(live_data, sequence_length, num_features):
    duplicated_data = np.tile(live_data, (sequence_length, 1))
    reshaped_data = duplicated_data.reshape(1, sequence_length, num_features)
    return reshaped_data


def live_data_prediction():
    latest_model_path = r'C:\Users\mmms\Desktop\BOT\model_folder\EURUSD5.h5'

    if latest_model_path:
        print(f'Using the latest model: {latest_model_path}')
        live_data = get_api_data()

        if live_data:
            live_data = np.array(live_data)
            scaler_filename = latest_model_path.replace('.h5', '_scaler.pkl')
            scaler = joblib.load(scaler_filename)
            normalized_live_data = scaler.transform(live_data)
            sequence_length = 5
            num_features = len(live_data[0])
            reshaped_live_data = reshape_live_data(normalized_live_data, sequence_length, num_features)
            prediction = make_predictions(latest_model_path, reshaped_live_data)
            create_file(prediction)
            #print(f'Predictions for the next five minutes: {prediction}')
            return prediction[0]


live_data_prediction()
