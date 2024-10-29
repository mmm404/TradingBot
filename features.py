import pandas as pd
import logging
import ta  # Technical Analysis library

class Indicators:
    def __init__(self, df, sma_short, sma_long, ema_short, ema_long, rsi_period, atr_period, stoch_period):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.stoch_period = stoch_period
        self.df = df

    def calculate_indicators(self, df: pd.DataFrame):
        try:
            # Calculate Simple Moving Averages (SMA)
            df[f'SMA_{self.sma_short}'] = ta.trend.SMAIndicator(df['close'], window=self.sma_short).sma_indicator()
            df[f'SMA_{self.sma_long}'] = ta.trend.SMAIndicator(df['close'], window=self.sma_long).sma_indicator()
            
            # Calculate Exponential Moving Averages (EMA)
            df[f'EMA_{self.ema_short}'] = ta.trend.EMAIndicator(df['close'], window=self.ema_short).ema_indicator()
            df[f'EMA_{self.ema_long}'] = ta.trend.EMAIndicator(df['close'], window=self.ema_long).ema_indicator()
            
            # Calculate Relative Strength Index (RSI)
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
            
            # Calculate Moving Average Convergence Divergence (MACD)
            macd = ta.trend.MACD(df['close'], window_slow=self.ema_long, window_fast=self.ema_short, window_sign=9)
            df['MACD'] = macd.macd()
            df['Signal'] = macd.macd_signal()
            
            # Calculate Bollinger Bands (20-period moving average with 2 standard deviations)
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['Upper_Band'] = bollinger.bollinger_hband()
            df['Lower_Band'] = bollinger.bollinger_lband()
            
            # Calculate Average True Range (ATR)
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.atr_period).average_true_range()
            
            # Calculate Stochastic Oscillator
            stochastic = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=self.stoch_period, smooth_window=3)
            df['Stochastic'] = stochastic.stoch()
            
            # Calculate Average Directional Index (ADX)
            df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
            
            # Manually calculate Heikin Ashi (HA) candlesticks
            df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
            df['HA_Open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna((df['open'] + df['close']) / 2)
            df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
            df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
            
            # Fill any missing data and drop NaN rows
            df.ffill(inplace=True)
            df.dropna(inplace=True)
            df.sort_index(inplace=True)


            # Save the DataFrame as a CSV file
            df.to_csv('calculated_indicators.csv', index=False)
            logging.info("DataFrame saved as 'calculated_indicators.csv'")

            # Return the dataframe with all calculated indicators
            return df
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            raise
