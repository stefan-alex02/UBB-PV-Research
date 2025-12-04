import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import TIMESTAMP_NAMES


def load_raw_data(file_path, target_col='AC_POWER'):
    """Loads and performs basic cleaning. Returns dataframe."""

    try:
        df = pd.read_csv(file_path)
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    for ts_name in TIMESTAMP_NAMES:
        if ts_name in df.columns:
            timestamp = pd.to_datetime(df[ts_name], errors='coerce')
            df["Minute"] = timestamp.dt.minute
            df["Hour"] = timestamp.dt.hour
            df["Day"] = timestamp.dt.day
            df["Month"] = timestamp.dt.month
            df["Year"] = timestamp.dt.year
            df = df.drop(columns=[ts_name])
            break

    df = df.fillna(method='ffill').fillna(method='bfill')
    target_col = target_col if target_col in df.columns else df.columns[-1]
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]

    return df.values.astype(np.float32), df.columns, dataset_name


def create_lags(data, n_in=3):
    """
    Transforms 2D array into 2D array with lagged features.
    Input: (N, F) -> Output: (N', F * (n_in + 1))
    """
    n_vars = data.shape[1]
    cols = []
    for i in range(n_in, 0, -1):
        cols.append(np.roll(data, i, axis=0))
    cols.append(data)

    agg = np.column_stack(cols)
    return agg[n_in:, :]


def create_sequences(data, seq_len=24):
    """
    Transforms 2D array into 3D array for LSTM.
    Input: (N, F) -> Output: X=(N', seq_len, F), y=(N', 1)
    Assumes the LAST column is the target.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len, :])
        y.append(data[i + seq_len, -1])

    return np.array(X), np.array(y).reshape(-1, 1)


def process_fold_data(X_train_raw, X_test_raw, method='standard', seq_len=3):
    """
    Scales and Transforms data based on method.
    Prevents leakage by fitting scaler ONLY on train.
    """
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(X_train_raw)
    test_scaled = scaler.transform(X_test_raw)

    if method == 'standard':
        # X is everything except last column, y is last column
        X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1:]
        X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1:]
        return X_train, y_train, X_test, y_test, scaler

    elif method == 'lagged':
        train_lagged = create_lags(train_scaled, n_in=seq_len)
        test_lagged = create_lags(test_scaled, n_in=seq_len)

        # Split X and y (Last column is y(t))
        X_train, y_train = train_lagged[:, :-1], train_lagged[:, -1:]
        X_test, y_test = test_lagged[:, :-1], test_lagged[:, -1:]
        return X_train, y_train, X_test, y_test, scaler

    elif method == 'sequence':
        X_train, y_train = create_sequences(train_scaled, seq_len)
        X_test, y_test = create_sequences(test_scaled, seq_len)
        return X_train, y_train, X_test, y_test, scaler


def create_dummy_csv(csv_path):
    dates = pd.date_range(start='1/1/2020', periods=2000, freq='h')
    df = pd.DataFrame({
        'DATE_TIME': dates,
        'Irradiance': np.abs(np.sin(np.linspace(0, 100, 2000)) * 1000 + np.random.normal(0, 50, 2000)),
        'Temperature': 25 + np.random.normal(0, 5, 2000),
        'Humidity': 50 + np.random.normal(0, 10, 2000),
        'AC_POWER': np.abs(np.sin(np.linspace(0, 100, 2000)) * 50 + np.random.normal(0, 2, 2000))
    })
    df.to_csv(csv_path, index=False)
