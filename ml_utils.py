import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# FINANCIAL FEATURES

# Calculats the Relative Strenght Index (RSI)
def calc_rsi(df, horizons=[14]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"RSI_{horizon}"
        new_df[name] = ta.rsi(new_df['Close'], length=horizon)
    return new_df

# Calculates the Simple Moving Average (SMA)
def calc_sma(df, horizons=[5,10,20,50,100]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"SMA_{horizon}"
        new_df[name] = ta.sma(new_df['Close'], length=horizon)
    return new_df
        
# Calculates the Exponential Moving Average (EMA)
def calc_ema(df, horizons=[20,50,100]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"EMA_{horizon}"
        new_df[name] = ta.ema(new_df['Close'], length=horizon)
    return new_df

# Calculates Moving Average Convergence/Divergence (MACD)
def calc_macd(df, signal=True):
    new_df = df.copy()
    new_df['MACD'] = ta.macd(new_df['Close'])['MACD_12_26_9']
    if (signal):
        new_df['MACD_Signal'] = ta.macd(new_df['Close'])['MACDs_12_26_9']
    return new_df

# Calculates upper and lower bollinger bands
def calc_bollinger_bands(df, horizons=[20]):
    new_df = df.copy()
    for horizon in horizons:
        bbands = ta.bbands(new_df['Close'], length=horizon)
        upper = f"BBU_{horizon}_2.0"
        lower = f"BBL_{horizon}_2.0"
        new_df[upper] = bbands[upper]
        new_df[lower] = bbands[lower]
    return new_df

# Calculates momentum
def calc_momentum(df, horizons=[10]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"momentum_{horizon}"
        new_df[name] = ta.mom(new_df['Close'], length=horizon)
    return new_df

#Calsulates Average True Range (ATR)
def calc_atr(df, horizons=[14]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"ATR_{horizon}"
        new_df[name] = ta.atr(new_df['High'], new_df['Low'], new_df['Close'], length=horizon)
    return new_df

def calc_all_default(df):
    new_df = df.copy()
    new_df = calc_rsi(new_df)
    new_df = calc_sma(new_df)
    new_df = calc_ema(new_df)
    new_df = calc_macd(new_df)
    new_df = calc_bollinger_bands(new_df)
    new_df = calc_momentum(new_df)
    new_df = calc_atr(new_df)
    return new_df

# UTILITIS FUNCTIONS

def directional_accuracy(y_true, y_pred, plot=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    true_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)

    true_dir = np.sign(true_diff)
    pred_dir = np.sign(pred_diff)

    correct = np.sum(true_dir == pred_dir)
    total = len(true_dir)
    
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(true_dir, label='Direzione Reale', marker='o')
        plt.plot(pred_dir, label='Direzione Predetta', marker='x')
        plt.title('Confronto Direzione Reale vs Predetta')
        plt.xlabel('Indice')
        plt.ylabel('Direzione (-1 = ↓, 0 = =, 1 = ↑)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return correct / total * 100

def model_scores(y_true, y_pred, verbose=True):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.clip(np.array(y_true), 1e-8, None))) * 100
    da = directional_accuracy(y_true, y_pred, plot=False)
    r2 = r2_score(y_true, y_pred)

    results = {
        'MAE': mae, # Mean Absolute Error
        'RMSE': rmse, # Root Mean Squared Error
        'MAPE (%)': mape, # Mean Absolute Percentage Error
        'DA': da, # Directional Accuracy
        'R²': r2 # R2 Score
    }

    if verbose:
        for k, v in results.items():
            print(f'{k}: {v:.4f}')

    return results

def plot_predictions(y_full, y_pred_train, y_pred_test, title, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    plt.plot(y_full, label='Real Values', color='blue')
    plt.plot(y_pred_train, label='Train Predictions', color='green')
    plt.plot(y_pred_test, label='Test Predictions', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()  
    
def create_time_windows(input, target, window_size=10):
    X = []
    y = []
    for i in range(len(input) - window_size):
        X.append(input[i:(i+window_size), :])
        y.append(target[i + window_size, 0])
    X = np.array(X)
    y = np.array(y)
    input_size = (X.shape[1], X.shape[2])
    return X, y, input_size

def train_test_split_timeWindows(input, target, window_size=10, train_size_pct=0.8):
    X, y, input_size = create_time_windows(input, target, window_size=window_size)
    train_size = int(len(X) * train_size_pct)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test, input_size, train_size

class DummyScaler:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array(X)
    
    def fit_transform(self, X):
        return np.array(X)
    
    def inverse_transform(self, X):
        return np.array(X)

class Data_Robust_Scaler:
    def __init__(self, input, target, features_to_ignore=[], dummy_scaler_for_target=False):
        self.input = input.copy()
        self.target = target.copy()
        
        features_to_scale = [f for f in self.input.columns if f not in features_to_ignore]
        
        X_scaler = ColumnTransformer(
            transformers=[
                ('scaler', RobustScaler(), features_to_scale),
                ('passthrough', 'passthrough', features_to_ignore)
            ]
        )
        
        if dummy_scaler_for_target:
            y_scaler = DummyScaler()
        else:
            y_scaler = RobustScaler()
        
        scaled_X = X_scaler.fit_transform(input)
        scaled_y = y_scaler.fit_transform(target)
        
        self.X_scaler = X_scaler
        self.y_scaler = y_scaler
        
        self.scaled_X = scaled_X
        self.scaled_y = scaled_y
        
    def get_scaled_data(self):
        return self.scaled_X, self.scaled_y
    
    def inverse_scale_output(self, y_output):
        return self.y_scaler.inverse_transform(y_output)
    
    def get_dataframe_results(self, y_train_pred, y_test_pred, train_size, window_size):
        train_results = self.target.iloc[:train_size, :]
        test_results = self.target.iloc[(train_size+window_size):, :]
        
        train_results['pred'] = self.inverse_scale_output(y_train_pred)
        test_results['pred'] = self.inverse_scale_output(y_test_pred)
        
        self.train_results = train_results
        self.test_results = test_results
        
        return train_results, test_results
    
    def get_test_results_with_forecast(self, scaled_forecast):
        forecast_window = len(scaled_forecast)
        forecast = self.inverse_scale_output(scaled_forecast)
        self.test_results['forecast'] = pd.NA
        self.test_results['forecast'].iloc[:forecast_window] = (
            forecast.reshape(forecast_window,)
        )
        return self.test_results
    
    def print_scores(self):
        print("Train-Set Score")
        model_scores(self.train_results['Close'], self.train_results['pred'])
        print("\nTest-Set Score")
        model_scores(self.test_results['Close'], self.test_results['pred'])
        
    def plot_results(self, title='Predictions', figsize=(16,8)):
        plot_predictions(self.target, self.train_results['pred'], self.test_results['pred'], title=title, figsize=figsize)
        
class LSTMModel:
    def __init__(self, input_shape, lstm_units = [100, 50], dense_units = [25], activation='relu', dropout=0.2, binary_classification=False):
        model = Sequential()
    
        # First LSTM layer
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=lstm_units[0], return_sequences=True))
        if (dropout):
            model.add(Dropout(dropout))
        
        for u in lstm_units[1:]:
            rs = (u != lstm_units[-1]) #return_sequences is False in the last LSTM layer.
            model.add(LSTM(units=u, return_sequences=rs))
            if (dropout):
                model.add(Dropout(dropout))
                
        for d in dense_units:
            model.add(Dense(units=d, activation=activation))
            
        if binary_classification:
            model.add(Dense(units=1, activation='sigmoid')) # Binary output
        else:
            model.add(Dense(units=1))
        
        if binary_classification:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer='adam', loss='mean_squared_error')
        
        #Saving the model
        self.model = model
        
    def fit(self, X_train, y_train, epochs=20, batch_size=8, verbose=1, set_early_stopping=True, validation_data=None):
        if validation_data is None:
            monitor = 'loss'
        else:
            monitor = 'val_loss'
        
        if (set_early_stopping):
            early_stopping = [EarlyStopping(
                monitor=monitor,
                patience=25,
                restore_best_weights=True
            )]
        else:
            early_stopping = []
        
        if validation_data is None:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=early_stopping,
                verbose=verbose
            )
        else:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=early_stopping,
                validation_data=validation_data,
                verbose=verbose
            )
        
        self.history = history
        
        return history
    
    def plot_loss(self):
        plt.figure(figsize=(16, 8))
        plt.semilogy(self.history.history['loss'], label='Loss Train')
        if (self.history.history['val_loss']):
            plt.semilogy(self.history.history['val_loss'], label='Loss Test')
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def predict(self, input, scaler=None):
        if scaler is None:
            pred = self.model.predict(input)
        else:
            # TO TEST
            pred = self.model.predict(input)
            pred = scaler.inverse_transform(pred)
        
        return pred
    
        
def forecast_autoregressive(model, initial_window, n_steps):
    window = initial_window
    predictions = []

    for _ in range(n_steps):
        next_pred = model.predict([window])[0] #Single prediction
        predictions.append(next_pred)
        
        window = window[1:]
        window = np.append(window, np.array([next_pred])).reshape(window.shape[0]+1, window.shape[1])

    return np.array(predictions)

def next_weekday_encoding(sin_val, cos_val):
    angle = np.arctan2(sin_val, cos_val)

    next_angle = angle + (2 * np.pi / 7)

    next_angle = (next_angle + np.pi) % (2 * np.pi) - np.pi

    next_sin = np.sin(next_angle)
    next_cos = np.cos(next_angle)
    return next_sin, next_cos

WEEKDAY_ENCODINGS = [
    (np.sin(2 * np.pi * i / 7), np.cos(2 * np.pi * i / 7))
    for i in range(7)
]

def next_weekday_encoding_by_lookup(sin_val, cos_val, tol=1e-8, encoding=WEEKDAY_ENCODINGS):
    for i, (s, c) in enumerate(encoding):
        if np.isclose(sin_val, s, atol=tol) and np.isclose(cos_val, c, atol=tol):
            next_index = (i + 1) % 7  # torna a lunedì dopo domenica
            return WEEKDAY_ENCODINGS[next_index]
    raise ValueError("Coppia (sin, cos) non riconosciuta come giorno valido.")

def forecast_autoregressive_circularDayEncription(model, initial_window, n_steps):
    window = initial_window
    predictions = []

    for _ in range(n_steps):
        next_pred = model.predict(window)[0] #Single prediction
        next_sin, next_cos = next_weekday_encoding_by_lookup(window[0][-1][1], window[0][-1][2])
        new_elem = [next_pred[0], next_sin, next_cos]
        # print(new_elem)
        # print(next_pred)
        # print(next_pred.shape)
        predictions.append(next_pred)
        
        temp = np.array(window[0][1:])
        # print(temp.shape)
        window[0] = np.append(temp, new_elem).reshape(temp.shape[0]+1, temp.shape[1])

    return np.array(predictions)