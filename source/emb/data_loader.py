import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_preprocess(path: str, timestamp_col: str, normalization: str) -> tuple[np.ndarray, list[str]]:
    """
    Load CSV, drop timestamp column, and normalize feature columns.

    Returns
    -------
    data : np.ndarray  shape (N, F)
    feature_names : list[str]
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != timestamp_col]
    X = df[feature_cols].values.astype(np.float32)

    if normalization == "standard":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalization == "minmax":
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}. Choose 'standard', 'minmax', or 'none'.")

    print(f"[Data] Loaded {X.shape[0]} rows × {X.shape[1]} features from '{path}'.")
    print(f"[Data] Normalization: {normalization}")
    return X, feature_cols


def build_windows(X: np.ndarray, k: int) -> np.ndarray:
    """
    Slide a window of size k over the rows and concatenate each window into one vector.

    Parameters
    ----------
    X : np.ndarray  shape (N, F)
    k : int         window size

    Returns
    -------
    windows : np.ndarray  shape (N - k + 1, k * F)
    """
    N, F = X.shape
    if k > N:
        raise ValueError(f"Window size k={k} exceeds dataset length {N}.")

    n_windows = N - k + 1
    windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(k, F))
    windows = windows.reshape(n_windows, k * F)
    print(f"[Windows] k={k} → {n_windows} windows of dimension {k * F}.")
    return windows.astype(np.float32)
