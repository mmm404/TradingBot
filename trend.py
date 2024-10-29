import pandas as pd  # For data manipulation
import logging  # For error handling and event logging
import MetaTrader5 as mt5  # To interact with MetaTrader 5 platform
import ta  # Technical Analysis library

class Trend:
    def __init__(self, df: pd.DataFrame):
        """
        Constructor to initialize the Trend class.
        This class uses technical indicators to detect market trends from a DataFrame containing price data.
        Args:
            df (pd.DataFrame): DataFrame with price data (open, high, low, close, etc.)
        """
        self.df = df  # Store the input DataFrame containing historical price data
        self.df = self.calculate_indicators(df)  # Apply technical indicators to the DataFrame

    @staticmethod
    def calculate_indicators(df: pd.DataFrame):
        """
        Calculate necessary technical indicators using the ta library.
        Args:
            df (pd.DataFrame): DataFrame with price data.
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        try:
            # Calculate Simple Moving Averages (SMA)
            df['SMA_3'] = ta.trend.SMAIndicator(df['close'], window=3).sma_indicator()
            df['SMA_10'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()

            # Calculate Relative Strength Index (RSI)
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=12).rsi()

            # Calculate MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['Signal'] = macd.macd_signal()

            # Remove rows with NaN values to avoid errors during trend evaluation
            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise

    @staticmethod
    def get_trend(symbol: str):
        """
        Static method to detect the current market trend (uptrend, downtrend, or sideways) for a given symbol
        on a 1-minute timeframe, using Simple Moving Averages (SMA), Relative Strength Index (RSI), and MACD.
        Args:
            symbol (str): The trading symbol (e.g., 'EURUSD') to analyze.
        Returns:
            str: Detected market trend ('uptrend', 'downtrend', 'sideways', or 'error' in case of failure).
        """
        try:
            # Fetch the last 50 bars of 1-minute data for the specified symbol from MetaTrader 5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
            df = pd.DataFrame(rates)  # Convert the rate data into a pandas DataFrame
            
            if df.empty:  # Check if data retrieval failed (empty DataFrame)
                logging.error(f"No data fetched for {symbol}")  # Log error if no data is retrieved
                return "error"  # Return error signal
            
            # Calculate technical indicators
            df = Trend.calculate_indicators(df)
            
            # Optimize SMA, RSI, and MACD trend detection logic for faster detection under 1-minute timeframe:
            # Trend determination: If the most recent SMA, RSI, and MACD values meet the criteria:
            SMA_3 = df['SMA_3'].iloc[-1]  # Latest 5-period SMA value
            sma_10 = df['SMA_10'].iloc[-1]  # Latest 10-period SMA value
            rsi = df['RSI'].iloc[-1]  # Latest RSI value
            macd = df['MACD'].iloc[-1]  # Latest MACD value
            signal = df['Signal'].iloc[-1]  # Latest MACD signal value
            
            # Uptrend: SMA_3 crosses above SMA_10, RSI is above 50, and MACD is above signal
            if SMA_3 > sma_10 and rsi > 30 and rsi < 60 and macd > signal:
                return "uptrend"
            # Downtrend: SMA_3 crosses below SMA_10, RSI is below 50, and MACD is below signal
            elif SMA_3 < sma_10 and rsi < 70 and rsi >50 and macd < signal:
                return "downtrend"
            # Sideways market: No clear direction (SMA_3 and SMA_10 are close, RSI near neutral, MACD and Signal are close)
            else:
                return "sideways"
        except Exception as e:
            # In case of any errors during the trend detection process, log the error and return 'error'
            logging.error(f"Error getting trend for {symbol}: {e}")
            return "error"  # Return error to signal failure
