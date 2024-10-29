import MetaTrader5 as mt5
import json
import logging
import pandas as pd
import os
from typing import Optional

# Configure logging
def setup_logging(log_path: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class MT5Interface:
    def __init__(self):
        self.config = self.load_config()
        if self.config is None:
            self.config = self.default_config()
        self.symbols = self.config['symbols']
        self.timeframe = self.get_timeframe(self.config['timeframe'])
        self.lot = self.calculate_lot_size()  # Dynamically calculate the lot size
        self.connected = False

    def load_config(self):
        """Load the JSON configuration file."""
        try:
            with open(r'C:/Users/peter/OneDrive/Desktop/python/Mainbot/logins.json', 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.error("Config file not found.")
            return None
        except json.JSONDecodeError:
            logging.error("Error decoding JSON config file.")
            return None

    def default_config(self):
        """Return a default configuration in case loading fails."""
        return {
            "username": os.getenv('MT5_USERNAME', 'your_username'),
            "password": os.getenv('MT5_PASSWORD', 'your_password'),
            "server": os.getenv('MT5_SERVER', 'your_server'),
            "symbols": ["EURUSD"],
            "timeframe": "M1"
        }

    def get_timeframe(self, timeframe_str):
        """Map timeframe strings to MetaTrader 5 constants."""
        timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        return timeframes.get(timeframe_str, mt5.TIMEFRAME_M15)

    def connect(self):
        """Connect to MetaTrader 5 with the account credentials."""
        uname = int(self.config['username'])
        pword = str(self.config['password'])
        trading_server = str(self.config['server'])
        filepath = str(self.config.get('path', ''))
        
        try:
            if mt5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
                logging.info("Trading Bot Starting")
                if mt5.login(login=uname, password=pword, server=trading_server):
                    logging.info("Trading Bot Logged in and Ready to Go!")
                    self.connected = True
                    return True
                else:
                    logging.error("MT5 Initialization Failed")
                    return False
        except Exception as e:
            logging.error(f"Error initializing MT5: {e}")
            return False

    def calculate_lot_size(self):
        """Calculate dynamic lot size based on account balance and risk percentage."""
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Unable to fetch account information.")
            return 0.1  # Default lot size if account info cannot be fetched
        
        # Use 1% risk per trade (adjust this to suit your risk management)
        risk_percentage = 1
        balance = account_info.balance
        lot_size = (balance * risk_percentage) / 1000  # Adjust risk management formula as necessary
        logging.info(f"Calculated lot size: {lot_size}")
        return max(0.01, lot_size)  # Ensure minimum lot size is not too small

    def place_order(self, order_type, symbol, volume, price, stop_loss=None, take_profit=None):
        """Place a market order with optional stop loss and take profit."""
        try:
            sl_price = price - stop_loss if order_type == 'buy' else price + stop_loss
            tp_price = price + take_profit if order_type == 'buy' else price - take_profit
            logging.debug(f"Placing order: {order_type} for {symbol} at price: {price}, SL: {sl_price}, TP: {tp_price}, Lot: {volume}")
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_price if stop_loss else 0,
                "tp": tp_price if take_profit else 0,
                "deviation": 10,
                "magic": 234000,
                "comment": "Python script open",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            logging.info(f"Order request: {request}")
            result = mt5.order_send(request)
            if result is None:
                logging.error("Order send failed: result is None")
            elif result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed, retcode={result.retcode}")
                logging.error(f"Order result: {result}")
            else:
                logging.info(f"Order placed successfully, {result}")
        except Exception as e:
            logging.error(f"Exception in place_order: {e}")

    def apply_trailing_stop(self, position, trailing_stop=0.1):
        """Apply a trailing stop to an existing position."""
        try:
            current_price = mt5.symbol_info_tick(position.symbol).ask if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).bid
            new_sl_price = current_price - trailing_stop if position.type == mt5.ORDER_TYPE_BUY else current_price + trailing_stop
            
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl_price,
                "tp": position.tp
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to apply trailing stop, retcode={result.retcode}")
            else:
                logging.info(f"Trailing stop applied successfully to position {position.ticket}")
        except Exception as e:
            logging.error(f"Exception in apply_trailing_stop: {e}")

    def manage_open_positions(self):
        """Manage open positions by checking if trailing stop should be applied."""
        positions = self.get_open_positions()
        if positions is None:
            logging.error("No open positions found")
            return
        
        for position in positions:
            self.apply_trailing_stop(position, trailing_stop=0.1)  # Adjust the trailing stop value as needed

    def get_open_positions(self):
        """Get currently open positions."""
        try:
            positions = mt5.positions_get()
            return positions
        except Exception as e:
            logging.error(f"Error getting open positions: {e}")
            return None

    def get_historical_data(self, symbol, num_bars):
        """Fetch historical data for a specific symbol."""
        try:
            rates = mt5.copy_rates_from_pos(symbol, self.timeframe, 0, num_bars)
            if rates is None:
                logging.error(f"Failed to get historical data for {symbol}")
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            logging.error(f"Exception fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_data(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M1, count: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch the latest historical data for a given symbol and return it as a DataFrame."""
        logging.info(f"Fetching latest data for symbol: {symbol}")
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                logging.error(f"No historical data found for {symbol}.")
                return pd.DataFrame()
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except mt5.MT5Error as e:
            logging.error(f"MetaTrader 5 error getting latest data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error getting latest data for {symbol}: {e}")
            return pd.DataFrame()

    def shutdown(self):
        """Shutdown MetaTrader 5 connection."""
        logging.info("Shutting down MT5 connection")
        mt5.shutdown()
        self.connected = False  # Set connection flag to False
