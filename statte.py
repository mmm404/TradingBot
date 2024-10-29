import time  # For sleep to control execution frequency in the trading loop
import numpy as np  # For numerical operations
import logging  # For logging information and errors
import pandas as pd  # For data manipulation using DataFrames
from features import Indicators  # Import technical indicators from custom features module
from trend import Trend  # Import trend detection logic from custom trend module
from intface import MT5Interface  # Import MT5 interface for interaction with MetaTrader 5
import MetaTrader5 as mt5  # MetaTrader 5 package for trading functionalities

# Class defining the trading strategy logic
class TradingStrategy:
    # Initialize the class with model and symbol parameters
    def __init__(self, model, symbol, df):
        self.model = model  # The trained LSTM model for price prediction
        self.symbol = symbol  # Forex symbol to trade on
        self.MT5 = MT5Interface()  # Initialize MT5 interface for trading functions
        self.df = df

    def generate_signal(self, df: pd.DataFrame) -> str:
        # Validate the input DataFrame to ensure it is not empty or incorrectly formatted
        if not isinstance(df, pd.DataFrame) or df.empty:
            logging.error("Invalid data format or empty data for signal generation")  # Log the error
            return 'hold'  # Return 'hold' if data is invalid
        try:
            # Use trend detection logic from the Trend class
            trend = Trend.get_trend(self.symbol)  # Get the trend for the symbol
        except Exception as e:
            logging.error(f"Error detecting trend: {e}")  # Log any error during trend detection
            return 'hold'  # Return 'hold' if there is a trend detection issue

        # Fetch current ask and bid prices from MT5
        ask_price = mt5.symbol_info_tick(self.symbol).ask  # Get the current ask price
        bid_price = mt5.symbol_info_tick(self.symbol).bid  # Get the current bid price
        try:
            # Prepare input data for the model (using the last 500 technical indicators)
            input_data = df[['SMA_3', 'SMA_10', 'RSI', 'MACD', 'ATR']].tail(500).values
            input_data = input_data.reshape((1, 500, input_data.shape[1]))  # Reshape data for LSTM input format (batch_size, timesteps, features)

            # Get the model's scaled prediction
            prediction = self.model.predict(input_data)

            # Log the predictions and prices
            logging.info(f"Detected trend: {trend}")
            logging.info(f"Model prediction: {prediction}")
            logging.info(f"ASK: {ask_price}")
            logging.info(f"BID: {bid_price}")
        # Catch key errors (e.g., if indicators column is missing) or value errors (e.g., shape mismatch)
        except KeyError as e:
            logging.error(f"Error generating signal: {e}")
            return 'hold'
        except ValueError as e:
            logging.error(f"Error generating signal: {e}")
            return 'hold'

        # Decision-making based on trend and predicted price
        if trend == 'uptrend' and prediction > ask_price:  # Buy signal if uptrend and predicted price > ask price
            logging.info("Buy signal generated")
            return 'buy'
        elif trend == 'downtrend' and prediction < bid_price:  # Sell signal if downtrend and predicted price < bid price
            logging.info("Sell signal generated")
            return 'sell'
        else:
            logging.info("Hold signal generated")  # Hold if conditions aren't met
            return 'hold'

    # Method to execute trades based on the signal
    def execute_trade(self, signal: str):
        try:
            # Fetch current ask and bid prices again from MT5
            ask_price = mt5.symbol_info_tick(self.symbol).ask
            bid_price = mt5.symbol_info_tick(self.symbol).bid
            volume = 1  # Example trade volume
            stop_loss = 1  # Example stop-loss in pips (should be calculated dynamically)
            take_profit = 2  # Example take-profit in pips (should be calculated dynamically)
            # Execute buy order if signal is 'buy'
            if signal == 'buy':
                self.MT5.place_order('buy', self.symbol, volume, ask_price, stop_loss, take_profit)
                logging.info("Buy order executed")
            # Execute sell order if signal is 'sell'
            elif signal == 'sell':
                self.MT5.place_order('sell', self.symbol, volume, bid_price, stop_loss, take_profit)
                logging.info("Sell order executed")
        except Exception as e:
            logging.error(f"Error executing trade: {e}")  # Log any errors during trade execution

    
    def run(self,df):
        while True:
            try:
                if not df.empty:  # If data was fetched successfully
                    signal = self.generate_signal(df)  # Generate trading signal
                    self.execute_trade(signal)  # Execute trade based on signal
                    
                    # Evaluate the model's prediction accuracy (RMSE) using the most recent data
                    X_test = df[['SMA_3', 'SMA_10', 'RSI', 'MACD', 'ATR']].values[-500:].reshape((1, 500, 5))
                    y_test = df['close'].values[-1:].reshape((1, 1))
                    rmse = self.model.evaluate(X_test, y_test)  # Calculate root mean squared error (RMSE)
                    logging.info(f"RMSE: {rmse}")
                    
                    # Manage open positions to ensure proper trade handling
                    self.MT5.manage_open_positions()
                    # Log that the trading loop is continuing
                    logging.info("Continuing trading loop...")
            except Exception as e:
                logging.error(f"Error in strategy run: {e}")  # Log any errors that occur in the loop
                
            time.sleep(60)  # Sleep for 60 seconds before repeating the loop
