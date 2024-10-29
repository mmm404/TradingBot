import MetaTrader5 as mt5
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class MT5Interface:
    def __init__(self):
        self.connected = False

    def connect(self):
        if not mt5.initialize():
            logging.error("MetaTrader 5 initialization failed")
            return False
        self.connected = True
        logging.info("MetaTrader 5 initialized successfully")
        return True

    def shutdown(self):
        if self.connected:
            mt5.shutdown()
            logging.info("MetaTrader 5 connection closed")
            self.connected = False

    def get_historical_data(self, symbol, timeframe, count=500):
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            logging.error(f"Failed to fetch historical data for {symbol}")
            return None
        return pd.DataFrame(rates)

    def train_model(self, data):
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data.set_index('time', inplace=True)
        X = data.drop(columns=['close'])
        y = data['close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        logging.info(f"Model trained with MSE: {mse}")
        return model

    def predict(self, model, data):
        prediction = model.predict(data)
        return prediction

# Example usage
mt5_interface = MT5Interface()
if mt5_interface.connect():
    historical_data = mt5_interface.get_historical_data('EURUSD', mt5.TIMEFRAME_H1, 1000)
    if historical_data is not None:
        model = mt5_interface.train_model(historical_data)
        latest_data = mt5_interface.get_historical_data('EURUSD', mt5.TIMEFRAME_H1, 1)
        if 'EURUSD' in latest_data:
            prediction = mt5_interface.predict(model, latest_data['EURUSD'])
            print(prediction)
            logging.info(f"Predicted price: {prediction}")
    mt5_interface.shutdown()
