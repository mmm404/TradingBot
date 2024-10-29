from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
import joblib
import logging
import numpy as np

class LSTMModel:
    def __init__(self, lstm_units=50, optimizer='adam', loss='mean_squared_error', dropout_rate=0.2):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Scaler for input features
        self.output_scaler = MinMaxScaler(feature_range=(0, 1))  # Separate scaler for output
        self.lstm_units = lstm_units
        self.optimizer = optimizer
        self.loss = loss
        self.dropout_rate = dropout_rate

    def create_model(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(units=self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(units=self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(units=1)
        ])
        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        try:
            X_train_scaled = self.scaler.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
            y_train_scaled = self.output_scaler.fit_transform(y_train)
            lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001)
            self.model.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[lr_callback])
        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

    def predict(self, X):
        if X.shape[1:] != (500, X.shape[2]):
            raise ValueError(f"Expected input shape (None, 500, {X.shape[2]}), found {X.shape}")
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        predictions_scaled = self.model.predict(X_scaled)
        predictions = self.output_scaler.inverse_transform(predictions_scaled)
        return predictions

    def save_model(self, model_path):
        self.model.save(model_path)
        joblib.dump(self.scaler, model_path + '_scaler.pkl')
        joblib.dump(self.output_scaler, model_path + '_output_scaler.pkl')

    def load_model(self, model_path):
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        self.scaler = joblib.load(model_path + '_scaler.pkl')
        self.output_scaler = joblib.load(model_path + '_output_scaler.pkl')

    def evaluate(self, X_test, y_test):
        try:
            X_test_scaled = self.scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)
            y_test_scaled = self.output_scaler.transform(y_test)
            predictions_scaled = self.model.predict(X_test_scaled)
            predictions = self.output_scaler.inverse_transform(predictions_scaled)
            rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            return rmse
        except Exception as e:
            logging.error(f"Error in evaluate method: {e}")
            raise
