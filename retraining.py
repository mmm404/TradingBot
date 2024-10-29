import logging  # For logging messages like info, warnings, and errors
import numpy as np  # For numerical operations and array manipulations
from features import Indicators  # Import the Indicators class to calculate technical indicators

# Class responsible for retraining the model with new historical data
class ModelRetrainer:
    # Constructor to initialize the retrainer with model, MT5 interface, and configuration
    def __init__(self, model, mt5, config, df):
        self.model = model  # The LSTM model that will be retrained
        self.mt5 = mt5  # Interface for interacting with MetaTrader 5 to fetch data
        self.config = config  # Configuration settings, such as paths, symbols, and other parameters
        self.df = df  # DataFrame with historical data

    # Method to retrain the model and save it
    def retrain_and_save_model(self, df, n_steps=500):
        try:
            # Check if historical data was successfully fetched
            if not df.empty:                
                X_train = []  # List to store input sequences for training
                y_train = []  # List to store target values for training

                # Create input and output sequences from historical data using technical indicators
                for i in range(len(df) - n_steps):
                    indicators = df.iloc[i:i+n_steps][['SMA_3', 'SMA_10', 'RSI', 'MACD', 'ATR']].values
                    X_train.append(indicators)  # Append 'n_steps' indicators as input
                    y_train.append(df['close'].iloc[i+n_steps])  # Append the 'n_steps+1' close price as output

                # Convert the input and output lists to NumPy arrays
                X_train = np.array(X_train).reshape(-1, n_steps, len(['SMA_3', 'SMA_10', 'RSI', 'MACD', 'ATR']))  # Reshape to (samples, timesteps, features)
                y_train = np.array(y_train).reshape(-1, 1)  # Reshape to (samples, 1)

                # Fit the scaler on the input training data (flattened to 2D)
                self.model.scaler.fit(X_train.reshape(-1, X_train.shape[2]))

                # Train the model with the new data
                self.model.train(X_train, y_train, epochs=10, batch_size=32)

                # Log that the model was retrained successfully
                logging.info("Model retrained with new data.")

                # Save the retrained model to the specified path in the config
                self.model.save_model(self.config['model_path'])

                # Log that the model was saved successfully
                logging.info(f"Model saved successfully at {self.config['model_path']}")
            else:
                # Log an error if the historical data could not be fetched
                logging.error("Failed to fetch historical data for retraining.")
        except Exception as e:
            # Log any errors that occur during the retraining process
            logging.error(f"Error during model retraining: {e}")
