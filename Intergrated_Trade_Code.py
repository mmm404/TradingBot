import os
import re
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import optimizers
import joblib
import time
from random import randint
from winotify import Notification, audio




data_li = []
time_li = []
csv_file_li = []

Symbols = ['EURUSD','USDJPY','USDCAD','AUDUSD','AUDCAD'] 
toaster = ToastNotifier()

def prepare_txt_data():
    print('Preparing data...')
    for file in os.listdir(os.path.join(os.getcwd(),'data_folder')):
        if file.split('.')[1] != 'csv':
            os.remove(os.path.join(os.path.join(os.getcwd(),'data_folder'),file))
        else:
            csv_file_li.append(os.path.join(os.path.join(os.getcwd(),'data_folder'),file))
    for i in csv_file_li:
        destination_path = r".\data_folder"
        data_path = os.path.join(os.path.join(os.getcwd(),'data_folder'),i)
        destination_path = os.path.join(destination_path,data_path.split("\\")[-1:][0].split(".")[0]+".txt")
        if not os.path.exists(destination_path):
            with open(destination_path,'w') as tfp:
                pass

        with open(data_path,'r') as dp:
            df = pd.DataFrame(pd.read_csv(dp))

        for item in df.values:
            row = str(item[0])
            with open(destination_path,'a') as fp:
                fp.write(row+"\n")
    print('done preparing')

def ordinalize_data(file_p):
    with open(file_p,'r') as fp:
        pattern = re.compile('\d{,4}-\d{,2}-\d{,2} \d{,2}:\d{,2}')
        for val in str(fp.read()).split('\n'):
            if len(str(val).split('\t')) == 6:
                time_row = str(val).split('\t')[0]
                data_row = [float(val) for val in str(val).split('\t')[1:]]  
                if pattern.match(time_row):
                    time_li.append(datetime.strptime(time_row, '%Y-%m-%d %H:%M').toordinal())
                    data_li.append(data_row)
    df = pd.DataFrame(data_li)
    df['Ordinal Time'] = time_li
    return df

def create_file(li):
    if not os.path.exists(r'.\prediction.csv','a'):
        with open(r'.\prediction.csv','w') as data_file:
            data_file.write(sum(li)/4)
    else:
        with open(r'.\prediction.csv','a') as data_file:
            data_file.write(sum(li)/4)


def normalize_data(df):
    scaler = MinMaxScaler()
    normalized_df = df.copy()
    features_to_normalize = df.columns[:-2]
    normalized_df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    return normalized_df, scaler



def create_lstm_model(dataframe, save_path=None):
    df = dataframe.copy()
    time_col = df.columns[-1]
    target_cols = df.columns[:-2]
    normalized_df, scaler = normalize_data(df)
    available_dp = len(normalized_df)
    target_cols_ind = list(map(int, target_cols))
    data = normalized_df.drop([time_col], axis=1).values[:, target_cols_ind]
    sequence_length = 5
    num_features = len(target_cols)
    generator = TimeseriesGenerator(data, data, length=sequence_length, batch_size=32)
    reshaped_data, reshaped_labels = generator[0]

    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(sequence_length, len(target_cols))))
    model.add(Dropout(0.2))
    model.add(LSTM(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_features, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    model.fit(generator, epochs=100, batch_size=32)

    if save_path:
        model.save(save_path)
        scaler_filename = save_path.replace('.h5', '_scaler.pkl')
        joblib.dump(scaler, scaler_filename)


def make_predictions(model_path, input_row):
    model = load_model(model_path)
    sequence_length = input_row.shape[1]
    num_features = input_row.shape[2]
    input_data = input_row.reshape(1, sequence_length, num_features)
    prediction = model.predict(input_data)[0]
    return prediction.tolist()



def get_api_data(param_index):
    api_key = 'ZASMYPGR0SQTETJN'
    url = "https://www.alphavantage.co/query"
    param_list = [{
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "EUR",
        "to_currency": "USD",
        "apikey": api_key
    },
    {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "JPY",
        "apikey": api_key
    },
    {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "USD",
        "to_currency": "CAD",
        "apikey": api_key
    },
    {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "AUD",
        "to_currency": "USD",
        "apikey": api_key
    },
    {
        "function": "CURRENCY_EXCHANGE_RATE",
        "from_currency": "AUD",
        "to_currency": "CAD",
        "apikey": api_key
    }]
    params = param_list[param_index]

    response = requests.get(url, params=params)
    data = response.json()

    currency_data = data.get('Realtime Currency Exchange Rate', {})
    exchange_rate = float(currency_data.get('5. Exchange Rate', 0))
    bid_price = float(currency_data.get('8. Bid Price', 0))
    ask_price = float(currency_data.get('9. Ask Price', 0))

    if exchange_rate == 0 and bid_price == 0 and ask_price == 0:
        print("Error: 'Realtime Currency Exchange Rate' not found in API response.")
        return None
    else:
        duplicated_value = ask_price
        features = [exchange_rate, bid_price, ask_price, duplicated_value]
        features = [float(value) for value in features]

        return [features]

def reshape_live_data(live_data, sequence_length, num_features):
    duplicated_data = np.tile(live_data, (sequence_length, 1))
    reshaped_data = duplicated_data.reshape(1, sequence_length, num_features)
    return reshaped_data

def live_data_prediction(parameter_index):
    latest_model_path = os.path.join(os.getcwd(),'model_folder',Symbols[parameter_index]+'.h5')

    if latest_model_path:
        print(f'Using the latest model: {latest_model_path}')
        live_data = get_api_data(param_index=parameter_index)

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
            return prediction[0]


def model_train():
    prepare_txt_data()
    for file in os.listdir(os.path.join(os.getcwd(),'data_folder')):
        if file.split('.')[1] == 'txt':
            model_alias = file.split('.')[0]+'.h5'
            full_path = os.path.join(os.path.join(os.getcwd(),'data_folder'),file)
            dataframe = ordinalize_data(full_path)
            save_model_path = os.path.join(os.path.join(os.getcwd(),'model_folder'),model_alias)
            create_lstm_model(dataframe, save_path=save_model_path)
            print(f'Done creating model name {model_alias}')
        
def windows_notifications(toast_msg):
    toast = Notification(app_id = "mmms fxbot",
                         title = "Trade Signal",
                         msg = toast_msg,
                         duration = "short")
    toast.set_audio(audio.default, loop=False)
    toast.show()
    
def equip_bot():
    if input("Create new models?[y/n] ").lower() == 'y':
        model_train()
        print('Done making models, preparing to predict ... \n')
    else:
        print("Using pre-created models in model folder ...\n")
       
def launch_bot():
    count = 0
    while True:
        if count != 5:
            windows_notifications(f'{Symbols[count]} {live_data_prediction(parameter_index=count)}')
        else:
            count = 0



equip_bot()
if input("Start Pedictions ?[y/n] ").lower() == 'y':
    print('Predicting next five minutes ')
    launch_bot()