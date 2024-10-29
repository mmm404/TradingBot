import asyncio  # Asynchronous I/O operations for non-blocking execution
import logging  # For logging errors, warnings, and info messages
import json  # For reading and writing JSON configuration files
from intface import MT5Interface  # Custom module to interface with MetaTrader 5 (MT5)
from modl import LSTMModel  # Custom module for the LSTM model used in predictions
from statte import TradingStrategy  # Custom module for the trading strategy
from retraining import ModelRetrainer  # Custom module for model retraining
import MetaTrader5 as MT5  # MT5 package for trading platform interaction
from features import Indicators

# Function to load configuration from a JSON file
def load_config(config_path='C:/Users/peter/OneDrive/Desktop/python/Mainbot/logins.json'):
    try:
        # Open and load the configuration file as a JSON object
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config  # Return the loaded configuration
    except FileNotFoundError:
        # Log an error if the configuration file is not found
        logging.error("Config file not found.")
        return None  # Return None to indicate failure
    except json.JSONDecodeError:
        # Log an error if the JSON file has invalid syntax or cannot be decoded
        logging.error("Error decoding JSON config file.")
        return None  # Return None to indicate failure

# Function to initialize logging to both a file and console
def setup_logging(log_path):
    # Configure the logging system to write to a file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_path)
    # Set up logging to the console (standard output)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)  # Set logging level to INFO
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # Define log format
    console.setFormatter(formatter)  # Apply the format to the console handler
    logging.getLogger('').addHandler(console)  # Add the console handler to the logger

# Main function to run the trading bot asynchronously
async def main():
    # Load the configuration file
    config = load_config()
    if not config:
        # Log an error and exit if the configuration failed to load
        logging.error("Failed to load configuration.")
        return
    # Set up logging based on the provided log path from the config
    setup_logging(config['log_path'])
    
    # Initialize the MT5 interface
    mt5 = MT5Interface()
    if not mt5.connect():
        # Log an error and exit if unable to connect to MetaTrader 5
        logging.error("Failed to connect to MetaTrader 5")
        return

    # Set the symbol to trade (in this case, 'XAUUSD' - Gold/USD)
    symbol = 'XAUUSD'
    # Fetch the latest 1000 bars of data for the symbol from MT5
    df= mt5.get_latest_data(symbol, MT5.TIMEFRAME_D1, count=1000)

    # Calculate technical indicators

    INDICATORS=Indicators(df, sma_short=3, sma_long=10, ema_short=10, ema_long=25, rsi_period=12, atr_period=10, stoch_period=14)
    df=INDICATORS.calculate_indicators(df)   

    # Initialize the LSTM model
    model = LSTMModel()
    model.create_model((500, 5))  # Create the model with the input shape (500 timesteps, 5 features)
    logging.info("Model created successfully")

    # Initialize the model retrainer for updating the LSTM model with new data
    retrainer = ModelRetrainer(model, mt5, config,df)
    logging.info("Training the model for the first time...")
    retrainer.retrain_and_save_model(df)  # Train and save the model initially
    
    # Load the trained model from the specified path in the config
    model.load_model(config['model_path'])
    logging.info(f"Model loaded successfully from {config['model_path']}")
    

    dff= mt5.get_latest_data(symbol, MT5.TIMEFRAME_M15, count=1000)
    dff=INDICATORS.calculate_indicators(df)
    # Initialize the trading strategy with the LSTM model and selected symbol
    strategy = TradingStrategy(model, symbol,dff)
    strategy.run(dff)  # Start the trading strategy

    # Asynchronous function to periodically retrain the model
    async def retrain_model_periodically():
        while True:
            await asyncio.sleep(60)  # Wait for 10 minutes before retraining
            logging.info("Retraining model with new data...")
            retrainer.retrain_and_save_model()  # Retrain and save the model
            model.load_model(config['model_path'])  # Reload the retrained model
            logging.info(f"Model reloaded after retraining from {config['model_path']}")
    
    # Create a background task for periodic model retraining
    asyncio.create_task(retrain_model_periodically())
    
    try:
        while True:
            try:
                if not dff.empty:
                    # Calculate technical indicators using the strategy
                    strategy.calculate_indicators(dff)
                    # Generate a trading signal based on the strategy
                    signal = strategy.generate_signal(dff)
                    # Execute the trade based on the generated signal
                    strategy.execute_trade(signal)
                    # Evaluate the model's accuracy using the Root Mean Squared Error (RMSE)
                    rmse = model.evaluate(dff[['SMA_3', 'SMA_10', 'RSI', 'MACD', 'ATR']].values[-500:].reshape((1, 500, 5)), df['close'].values[-1:].reshape((1, 1)))
                    logging.info(f"RMSE: {rmse}")
                    # Manage open positions based on the trading strategy
                    strategy.MT5.manage_open_positions()
            except Exception as e:
                # Log any errors that occur within the trading loop
                logging.error(f"Error in trading loop: {e}")
            # Wait for the update interval (from the config) before fetching new data and trading again
            await asyncio.sleep(config['update_interval'])
    except KeyboardInterrupt:
        # Log when the bot is interrupted and shut down
        logging.info("Shutting down the trading bot...")
    finally:
        # Ensure MT5 connection is properly shut down when the bot ends
        mt5.shutdown()

# Entry point to run the bot
if __name__ == "__main__":
    asyncio.run(main())  # Run the main function asynchronously
