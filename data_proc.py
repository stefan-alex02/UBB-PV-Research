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
            df[ts_name] = pd.to_datetime(df[ts_name], errors='coerce')
            df["Minute"] = df[ts_name].dt.minute
            df["Hour"] = df[ts_name].dt.hour
            df["DayOfYear"] = df[ts_name].dt.dayofyear
            df["Year"] = df[ts_name].dt.year

            df = df.sort_values(by=ts_name)
            df = df.set_index(ts_name)
            break

    df = df.fillna(method='ffill').fillna(method='bfill')
    feature_cols = [c for c in df.columns if c != target_col]
    df = df[feature_cols + [target_col]]

    return df, dataset_name


def load_unisolar_data(path, target_site_key):
    """
    Loads and merges UNISOLAR data for a specific SiteKey.
    """
    print(f"Loading Solar Data for SiteKey: {target_site_key}...")

    target_col = 'SolarGeneration'

    solar_path = os.path.join(path, 'Solar_Energy_Generation.csv')
    weather_path = os.path.join(path, 'Weather_Data_reordered_all.csv')

    # 1. Load Solar Data
    # We load everything first, but you could use chunksize if it were >1GB
    df_solar = pd.read_csv(solar_path)

    # Filter immediately to reduce memory usage
    df_solar = df_solar[df_solar['SiteKey'] == target_site_key].copy()

    if df_solar.empty:
        raise ValueError(f"No data found for SiteKey {target_site_key}")

    # Convert Timestamp
    df_solar['Timestamp'] = pd.to_datetime(df_solar['Timestamp'])

    # Identify which Campus this Site belongs to (to find matching weather)
    # We assume a Site doesn't move campuses, so we take the first value.
    target_campus_key = df_solar['CampusKey'].iloc[0]
    print(f"Site {target_site_key} is located on Campus {target_campus_key}. Loading corresponding weather...")

    # 2. Load Weather Data
    df_weather = pd.read_csv(weather_path)
    df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])

    # Filter Weather for the correct Campus
    df_weather = df_weather[df_weather['CampusKey'] == target_campus_key].copy()

    # 3. Merge Strategy
    # We perform a LEFT JOIN on 'Timestamp' and 'CampusKey'.
    # Left Join ensures we keep all Solar rows, even if weather is missing for a minute.
    df_merged = pd.merge(
        df_solar,
        df_weather,
        on=['Timestamp', 'CampusKey'],
        how='left'
    )

    # 4. Handle Missing Values
    # Solar Generation: Fill NaNs with 0 (as requested/night time logic)
    df_merged[target_col] = df_merged[target_col].fillna(0)

    # Weather: Do NOT fill with 0 (0°C is valid weather). Use Forward Fill.
    # This propagates the last known temperature to missing slots.
    weather_cols = [c for c in df_weather.columns if c not in ['Timestamp', 'CampusKey']]
    df_merged[weather_cols] = df_merged[weather_cols].fillna(method='ffill').fillna(method='bfill')

    # 5. Final Cleanup
    # Sort by time
    df_merged = df_merged.sort_values('Timestamp')

    df_merged['Minute'] = df_merged['Timestamp'].dt.minute
    df_merged['Hour'] = df_merged['Timestamp'].dt.hour
    df_merged['DayOfYear'] = df_merged['Timestamp'].dt.dayofyear
    df_merged['Year'] = df_merged['Timestamp'].dt.year

    # Set Index for time-series processing
    df_merged.set_index('Timestamp', inplace=True)

    # Drop IDs now that they are merged
    df_merged = df_merged.drop(columns=['CampusKey', 'SiteKey'], errors='ignore')

    feature_cols = [c for c in df_merged.columns if c != target_col]
    df_merged = df_merged[feature_cols + [target_col]]

    print(f"Merged Data Shape: {df_merged.shape}")
    return df_merged, "unisolar"


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
    Scales X and y independently, then applies time-series windowing.
    Assumes the LAST column of the input is the Target.
    """
    # 1. Initialize Independent Scalers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 2. Split Raw Data into Features vs Target
    # We assume the last column is the target (enforced in load_data)
    X_train_unscaled = X_train_raw[:, :-1]
    y_train_unscaled = X_train_raw[:, -1:]

    X_test_unscaled = X_test_raw[:, :-1]
    y_test_unscaled = X_test_raw[:, -1:]

    # 3. Fit & Transform
    # Fit scaler ONLY on training data to prevent leakage
    X_train_scaled = scaler_X.fit_transform(X_train_unscaled)
    y_train_scaled = scaler_y.fit_transform(y_train_unscaled)

    # Apply same transformation to test data
    X_test_scaled = scaler_X.transform(X_test_unscaled)
    y_test_scaled = scaler_y.transform(y_test_unscaled)

    # 4. Handle Processing Methods
    if method == 'standard':
        # No windowing needed, just return the scaled independent parts
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_X, scaler_y

    # For 'lagged' and 'sequence', we need to recombine X and y
    # because the history of y (past power) is an input feature for the future.
    train_combined = np.hstack([X_train_scaled, y_train_scaled])
    test_combined = np.hstack([X_test_scaled, y_test_scaled])

    if method == 'lagged':
        train_lagged = create_lags(train_combined, n_in=seq_len)
        test_lagged = create_lags(test_combined, n_in=seq_len)

        # Split again: All columns except last are features, Last is target(t)
        X_train, y_train = train_lagged[:, :-1], train_lagged[:, -1:]
        X_test, y_test = test_lagged[:, :-1], test_lagged[:, -1:]
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y

    elif method == 'sequence':
        X_train, y_train = create_sequences(train_combined, seq_len)
        X_test, y_test = create_sequences(test_combined, seq_len)
        return X_train, y_train, X_test, y_test, scaler_X, scaler_y


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
