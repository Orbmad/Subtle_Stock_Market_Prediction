# AUTHOR: Manuele D'Ambrosio :)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_ta as ta
import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# EVALUATION METRICS

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
        plt.plot(true_dir, label='Real Direction', marker='o')
        plt.plot(pred_dir, label='Predicted Direction', marker='x')
        plt.title('Real Direction vs Predicted Direction')
        plt.xlabel('Index')
        plt.ylabel('Direction (-1 = ↓, 0 = =, 1 = ↑)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return correct / total * 100

def model_scores(y_true, y_pred, verbose=True):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
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

def direction_momentum(df, horizons=[5, 10, 20, 60]):
    new_df = df.copy()
    for horizon in horizons:
        name = f"direction_mom_{horizon}"
        new_df[name] = new_df['direction'].rolling(horizon).sum() - (horizon/2)
    return new_df

def add_direction_features(df, horizons=[5, 10, 20, 60, 80, 100, 120, 140], intraday_delta=True):
    new_df = df.copy()
    
    if intraday_delta:
        new_df['delta'] = new_df['Close'] - new_df['Open']
    else:
        new_df['delta'] = new_df['Close'].diff()
        
    new_df['direction'] = (new_df['delta'] > 0).astype(int)
    new_df = direction_momentum(new_df, horizons)
    new_df['next'] = new_df['direction'].shift(-1)
    new_df = new_df.dropna()
    
    
# DATA SPLITS

def split_before_year(X, y, year):
    is_train = X.index.year < year
    X_train = X.loc[is_train]
    y_train = y.loc[is_train]
    X_val = X.loc[~is_train]
    y_val = y.loc[~is_train]
    return X_train, X_val, y_train, y_val

def split_train_test_pct(X, y, train_pct=0.8):
    train_size = int(len(X)*train_pct)
    X_train, y_train = X.iloc[:train_size, :], y.iloc[:train_size, :]
    X_test, y_test = X.iloc[train_size:, :], y.iloc[train_size:, :]
    return X_train, y_train, X_test, y_test

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
    

# MODELS

class LassoClassifier:
    def __init__(self, scaler=MinMaxScaler(), reg_intensity=1):
        C = 1/reg_intensity
        lasso = Pipeline([
            ('scaler', scaler),
            ('lasso', LogisticRegression(penalty='l1',
                                        solver='saga',
                                        C=C,
                                        max_iter=1000))
        ])
        self.model = lasso
        
    def fit(self, X, y):
        self.model.fit(X, y)
        self.X = X
        self.y = y
    
    def predict(self, input):
        pred = self.model.predict(input)
        return pred
    
    def plot_results(self, y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))
        
    def plot_features_analysis(self, figsize=(14, 7)):
        lasso = self.model.named_steps['lasso']
        coefs = lasso.coef_[0]
        selected = [i for i, w in enumerate(coefs) if w != 0]
        print('Selected features: ', selected)
        
        feature_names = self.X.columns if hasattr(self.X, 'columns') else [f"f{i}" for i in range(self.X.shape[1])]
        plt.figure(figsize=figsize)
        plt.bar(feature_names, coefs)
        plt.axhline(0, color='gray', linestyle='--')
        plt.xticks(rotation=90)
        plt.title("Feature importance (L1 coeffs)")
        plt.tight_layout()
        plt.show()
        
        return selected
        
    def plot_correlation_matrix(self, annot_kws={"size": 6}, cmap='coolwarm'):
        corr_mat = self.X.corr()
        sns.heatmap(corr_mat, annot=True, annot_kws=annot_kws, cmap=cmap)
        plt.title('Correlation Matrix')
        plt.show()
        print('Determinant: ', np.linalg.det(corr_mat))
        
class LSTMClassifier:
    def __init__(self, input_size, lstm_units, dense_units, dropout=None):
        lstm_model = Sequential()
        lstm_model.add(Input(shape=input_size))
        lstm_model.add(LSTM(units=lstm_units[0], return_sequences=True))
        for u in lstm_units[1:]:
                    rs = (u != lstm_units[-1]) #return_sequences is False in the last LSTM layer.
                    lstm_model.add(LSTM(units=u, return_sequences=rs))
                    if dropout is not None:
                        lstm_model.add(Dropout(dropout))
        for d in dense_units:
                    lstm_model.add(Dense(units=d, activation='relu'))
        lstm_model.add(Dense(units=1, activation='sigmoid'))
        
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = lstm_model
        
    def fit(self, X, y, scaler=MinMaxScaler(), epochs=20, batch_size=8, validation_data=None, early_stopping='val_loss'):
        scaled_X = scaler.fit_transform(X)
        
        if validation_data is None:
            monitor='loss'
        else:
            monitor='val_loss'
            
        if early_stopping is not None:
            es = [EarlyStopping(
                monitor=monitor,
                patience=20,
                restore_best_weights=True
            )]
        else:
            es = []

        history = self.model.fit(
            scaled_X, y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=es,
            validation_data=validation_data,
            verbose=True
        )
        self.history = history
        return history
    
    def predict(self, X):
        pred = self.model.predict(X)
        return pred
    
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
        
class RandomModel:
    def __init__(self, target):
        self.mean = target.mean()
        self.std = target.std()
        self.len = len(target)

    def random_predictions(self, len=None):
        if len is None:
            len = self.len
        rand_preds = np.random.normal(
            self.mean,
            self.std,
            len
        )
        return rand_preds
       
# TRADING UTILS

def gain(open_df, C, C_pred):
    O = open_df.reindex_like(C)
    CO_diff = C - O
    growth = C_pred > O
    decline = C_pred < O
    return CO_diff[growth].sum() - CO_diff[decline].sum()

def roi(open_df, C, C_pred):
    mean_open = open_df.reindex_like(C).mean()
    return gain(C, C_pred) / mean_open

def print_eval(X, y, model):
    preds = model.predict(X)
    print("Gain: {:.2f}$".format(gain(y, preds)))
    print(" ROI: {:.3%}".format(roi(y, preds)))
    
def dir_gain(delta_true_df, dir_pred):
    growth = dir_pred == 1
    decline = dir_pred == 0
    return delta_true_df[growth].sum() - delta_true_df[decline].sum()

def dir_roi(open_df, delta_true_df, dir_pred):
    mean_open = open_df.reindex_like(delta_true_df).mean()
    return gain(delta_true_df, dir_pred) / mean_open