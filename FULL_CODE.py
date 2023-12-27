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

data_li = []
time_li = []
csv_file_li = []

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
                data_row = [float(val) for val in str(val).split('\t')[1:]]  # Convert to float
                if pattern.match(time_row):
                    time_li.append(datetime.strptime(time_row, '%Y-%m-%d %H:%M').toordinal())
                    data_li.append(data_row)
    df = pd.DataFrame(data_li)
    df['Ordinal Time'] = time_li
    return df

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
    model.fit(generator, epochs=1, batch_size=32)

    if save_path:

        model.save(save_path)

        scaler_filename = save_path.replace('.h5', '_scaler.pkl')
        joblib.dump(scaler, scaler_filename)

    return model


def make_predictions(model_path, input_row):
    model = load_model(model_path)
    input_data = np.array(input_row).reshape(1, 4, len(input_row))

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

    exchange_rate = [data['Realtime Currency Exchange Rate']['5. Exchange Rate']]
    bid_price = [data['Realtime Currency Exchange Rate']['8. Bid Price']]
    ask_price = [data['Realtime Currency Exchange Rate']['9. Ask Price']]

    DataF1 = pd.DataFrame()
    DataF1['Exchange_rate'] = exchange_rate
    DataF1['Bid_price'] = bid_price
    DataF1['Ask_price'] = ask_price

def model_train():
    #prepare_txt_data()
    for file in os.listdir(os.path.join(os.getcwd(),'data_folder')):
        if file.split('.')[1] == 'txt':
            model_alias = file.split('.')[0]+'.h5'
            full_path = os.path.join(os.path.join(os.getcwd(),'data_folder'),file)
            dataframe = ordinalize_data(full_path)
            save_model_path = os.path.join(os.path.join(os.getcwd(),'model_folder'),model_alias)
            create_lstm_model(dataframe, save_path=save_model_path)
            print(f'Done creating model name {model_alias}')
        #prediction = make_predictions(save_model_path, ordinalize_data(file))



def launch_bot():
    model_train()

launch_bot()
