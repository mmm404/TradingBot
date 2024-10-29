import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from modl import LSTMModel
import logging
import queue

def preprocess_data(close_prices, window_size=60):
    X, y = [], []
    for i in range(window_size, len(close_prices)):
        X.append(close_prices[i-window_size:i])
        y.append(close_prices[i])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    print(X.shape,y.shape)
    return X, y

def train_and_evaluate_model(close_prices, result_queue):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    X, y = preprocess_data(close_prices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lstm_model = LSTMModel()
    lstm_model.create_model((X_train.shape[1], X_train.shape[2]))
    lstm_model.train(X_train, y_train, epochs=10, batch_size=32)

    predictions = lstm_model.predict(X_test)
    y_test = y_test.reshape(-1, 1)

    rmse = np.sqrt(np.mean((predictions - y_test)**2))
    logging.info(f'Root Mean Squared Error: {rmse}')

    buy_signals = predictions > lstm_model.buy_threshold
    true_buy_signals = y_test > lstm_model.buy_threshold

    buy_precision = precision_score(true_buy_signals, buy_signals, zero_division=0)
    buy_recall = recall_score(true_buy_signals, buy_signals, zero_division=0)
    buy_f1 = f1_score(true_buy_signals, buy_signals, zero_division=0)

    logging.info(f'Buy Precision: {buy_precision}')
    logging.info(f'Buy Recall: {buy_recall}')
    logging.info(f'Buy F1 Score: {buy_f1}')

    sell_signals = predictions < lstm_model.sell_threshold
    true_sell_signals = y_test < lstm_model.sell_threshold

    sell_precision = precision_score(true_sell_signals, sell_signals, zero_division=0)
    sell_recall = recall_score(true_sell_signals, sell_signals, zero_division=0)
    sell_f1 = f1_score(true_sell_signals, sell_signals, zero_division=0)

    logging.info(f'Sell Precision: {sell_precision}')
    logging.info(f'Sell Recall: {sell_recall}')
    logging.info(f'Sell F1 Score: {sell_f1}')

    result_queue.put((lstm_model, predictions, y_test))

def save_model(model, model_path):
    try:
        model.save_model(model_path)
        logging.info(f"Model saved successfully at {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

def load_model(model_path):
    try:
        model = LSTMModel()
        model.load_model(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def main():
    close_prices = np.random.rand(1000)

    result_queue = queue.Queue()
    train_and_evaluate_model(close_prices, result_queue)

    lstm_model, predictions, y_test = result_queue.get()

    save_model(lstm_model, "lstm_model.h5")

    loaded_model = load_model("lstm_model.h5")

    if loaded_model:
        rmse = loaded_model.evaluate(predictions, y_test)
        logging.info(f"Loaded model RMSE: {rmse}")

if __name__ == "__main__":
    main()