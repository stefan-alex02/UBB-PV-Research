from enum import Enum

import numpy as np
from pandas import DataFrame
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


class ProcessingMethod(Enum):
    STANDARD = 'standard'
    LAGGED = 'lagged'
    SEQUENCE = 'sequence'


def split_and_process_folds(df: DataFrame, k_folds: int,
                            method: ProcessingMethod | str, seq_len: int):
    """
    Generator that yields processed data for each fold.
    :return: fold index, X_train, y_train, X_test, y_test, scaler_y
    """

    method = method if isinstance(method, ProcessingMethod) else ProcessingMethod(method)
    kf = KFold(n_splits=k_folds, shuffle=False)

    # todo handle k fold == 1 case
    for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_data = df.iloc[train_idx]
        test_data = df.iloc[test_idx]

        X_train, y_train, X_test, y_test, _, scaler_y = process_fold_data(
            train_data.values.astype(np.float32),
            test_data.values.astype(np.float32),
            method=method,
            seq_len=seq_len
        )

        yield fold, X_train, y_train, X_test, y_test, scaler_y


def process_fold_data(X_train_raw, X_test_raw, method: ProcessingMethod, seq_len: int):
    """
    Scales X and y independently, then applies time-series windowing. Target column is assumed to be the last one.
    :return:
    """
    # Independent Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Last column is assumed to be the Target
    X_train_unscaled = X_train_raw[:, :-1] # todo inefficient copy
    y_train_unscaled = X_train_raw[:, -1:]

    X_test_unscaled = X_test_raw[:, :-1]
    y_test_unscaled = X_test_raw[:, -1:]

    # Fit & Transform
    X_train_scaled = scaler_X.fit_transform(X_train_unscaled)
    y_train_scaled = scaler_y.fit_transform(y_train_unscaled)

    X_test_scaled = scaler_X.transform(X_test_unscaled)
    y_test_scaled = scaler_y.transform(y_test_unscaled)

    # 4. Handle Processing Methods
    if method == ProcessingMethod.STANDARD:
        # No windowing needed, just return the scaled independent parts
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y

    train_combined = np.hstack([X_train_scaled, y_train_scaled])
    test_combined = np.hstack([X_test_scaled, y_test_scaled])

    if method == ProcessingMethod.LAGGED:
        train_lagged = create_lags(train_combined, n_in=seq_len)
        test_lagged = create_lags(test_combined, n_in=seq_len)

        # Split again: All columns except last are features, Last is target(t)
        X_train, y_train = train_lagged[:, :-1], train_lagged[:, -1:]
        X_test, y_test = test_lagged[:, :-1], test_lagged[:, -1:]
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y

    elif method == ProcessingMethod.SEQUENCE:
        X_train, y_train = create_sequences(train_combined, seq_len)
        X_test, y_test = create_sequences(test_combined, seq_len)
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y

    else:
        raise ValueError(f"Unknown processing method: {method}")


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
    Transforms 2D array into 3D array (for LSTM).
    Input: (N, F) -> Output: X=(N', seq_len, F), y=(N', 1)
    Assumes the LAST column is the target.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len, :])
        y.append(data[i + seq_len, -1])

    return np.array(X), np.array(y).reshape(-1, 1)
