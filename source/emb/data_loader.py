import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_preprocess(
    path: str,
    timestamp_col: str,
    normalization: str,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Load CSV, drop timestamp, normalize features, and keep the RAW (un-normalized)
    target column separately so that downstream class ranges and |delta y| are in physical units.

    Returns
    -------
    X            : np.ndarray (N, F)   normalized feature matrix (still includes target col)
    y_raw        : np.ndarray (N,)     raw, un-normalized target values
    feature_cols : list[str]           names of the F columns in X
    target_idx   : int                 index of target_col within feature_cols
    """
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c != timestamp_col]
    if target_col not in feature_cols:
        raise ValueError(f"Target column {target_col!r} not found in dataset columns {feature_cols}.")
    target_idx = feature_cols.index(target_col)

    X = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)   # raw target, before normalization

    if normalization == "standard":
        X = StandardScaler().fit_transform(X)
    elif normalization == "minmax":
        X = MinMaxScaler().fit_transform(X)
    elif normalization == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}. Choose 'standard', 'minmax', or 'none'.")

    print(f"[Data] Loaded {X.shape[0]} rows × {X.shape[1]} features from '{path}'.")
    print(f"[Data] Normalization: {normalization}  |  Target column: {target_col!r} (idx {target_idx})")
    return X.astype(np.float32), y_raw, feature_cols, target_idx


def build_windows(
    X: np.ndarray,
    y_raw: np.ndarray,
    k: int,
    mode: str,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of size k over the rows and concatenate each window into one vector,
    returning the per-window target y according to `mode`.

    Returns
    -------
    windows : np.ndarray (M, D)   M windows, each of dimension D
    y       : np.ndarray (M,)     raw target per window
    """
    N, F = X.shape
    if k > N:
        raise ValueError(f"Window size k={k} exceeds dataset length {N}.")

    if mode == "next_y":
        # Keep all columns
        n_full = N - k + 1
        windows = np.lib.stride_tricks.sliding_window_view(X, window_shape=(k, F))
        windows = windows.reshape(n_full, k * F)
        windows = windows[:-1]              # drop last window
        y = y_raw[k:]                       # aligned
        assert len(windows) == len(y), (len(windows), len(y))
        print(f"[Windows] mode='next_y'  k={k} -> {len(windows)} windows of dim {k * F} "
              f"(last window sacrificed). Target = next-step SolarGeneration.")

    elif mode == "latest_y":
        # Drop the target column from features
        feat_cols = [c for c in range(F) if c != target_idx]
        X_nt = X[:, feat_cols]
        F2 = X_nt.shape[1]
        n_full = N - k + 1
        windows = np.lib.stride_tricks.sliding_window_view(X_nt, window_shape=(k, F2))
        windows = windows.reshape(n_full, k * F2)
        y = y_raw[k - 1:]                   # latest timestep of each window
        assert len(windows) == len(y), (len(windows), len(y))
        print(f"[Windows] mode='latest_y'  k={k} -> {len(windows)} windows of dim {k * F2} "
              f"(target column excluded from features). Target = latest-step SolarGeneration.")

    else:
        raise ValueError(f"Unknown WINDOWING_MODE: {mode!r}. Choose 'next_y' or 'latest_y'.")

    return windows.astype(np.float32), y.astype(np.float32)


def discretize_target(y: np.ndarray, n_classes: int, low_fraction: float) -> tuple[np.ndarray, dict]:
    """
    Discretize target y into classes based on fractions of the MAXIMUM y value.

    Class 0      : y < low_fraction * y_max          (the large "low/night" class)
    Classes 1..n : the remaining range [low_fraction*y_max, y_max] split into
                   (n_classes - 1) equal-width bins.

    Returns
    -------
    classes    : np.ndarray (M,) int class labels in [0, n_classes-1]
    class_info : dict with 'y_max', 'low_threshold', 'bin_edges', 'class_ranges'
    """
    y_max = float(y.max())
    low_threshold = low_fraction * y_max

    n_high = n_classes - 1
    high_edges = np.linspace(low_threshold, y_max, n_high + 1)   # low_threshold ... y_max
    bin_edges = np.concatenate([[0.0], high_edges])             # [0, low_thr, ..., y_max]

    classes = np.digitize(y, bin_edges[1:-1], right=False)
    classes = np.clip(classes, 0, n_classes - 1).astype(int)

    class_ranges = []
    for c in range(n_classes):
        lo, hi = bin_edges[c], bin_edges[c + 1]
        count = int((classes == c).sum())
        class_ranges.append({"class": c, "low": float(lo), "high": float(hi), "count": count})

    class_info = {
        "y_max": y_max,
        "low_fraction": low_fraction,
        "low_threshold": float(low_threshold),
        "bin_edges": bin_edges.tolist(),
        "class_ranges": class_ranges,
        "n_classes": n_classes,
    }

    print(f"[Classes] y_max={y_max:.3f}  low_threshold={low_threshold:.3f} "
          f"({low_fraction*100:.0f}% of max)")
    for cr in class_ranges:
        print(f"          Class {cr['class']}: [{cr['low']:.2f}, {cr['high']:.2f})  n={cr['count']}")

    return classes, class_info
