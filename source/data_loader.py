import pandas as pd
from pandas import DataFrame


loading_methods: dict = {
    'single_csv': lambda **kwargs: load_single_csv(**kwargs),
    'merged_csvs': lambda **kwargs: load_csvs_filter_merge(**kwargs),
}


def load_single_csv(csv_path: str, **_) -> DataFrame:
    """
    Loads a single CSV file into a DataFrame.
    """
    return pd.read_csv(csv_path)


def load_csvs_filter_merge(solar_path: str, weather_path: str,
                           site_key_col: str, site_key: int,
                           campus_key_col: str, timestamp_col: str,
                           **_) -> DataFrame:
    """
    Loads two CSV files (Solar and Weather), filters them based on SiteKey and CampusKey,
    and merges them on Timestamp.
    """
    df_solar = pd.read_csv(solar_path)
    df_solar = df_solar[df_solar[site_key_col] == site_key].copy()

    if df_solar.empty:
        raise ValueError(f"No data found for {site_key_col}: {site_key}")

    df_solar[timestamp_col] = pd.to_datetime(df_solar[timestamp_col])

    filtered_campus = df_solar[campus_key_col].iloc[0]

    # Load weather data
    df_weather = pd.read_csv(weather_path)
    df_weather[timestamp_col] = pd.to_datetime(df_weather[timestamp_col])

    # Filter Weather for the correct Campus
    df_weather = df_weather[df_weather[campus_key_col] == filtered_campus].copy()

    # Merge strategy using LEFT JOIN
    return pd.merge(
        df_solar,
        df_weather,
        on=[timestamp_col,
            campus_key_col  # todo remove if unnecessary
            ],
        how='left'
    )


def drop_columns(df: DataFrame, cols_to_drop: list = None) -> DataFrame:
    """
    Drops specified columns from the DataFrame.
    """
    return df.drop(columns=cols_to_drop, errors='ignore')


def make_timestamp_index(df: DataFrame, timestamp_col: str) -> DataFrame:
    """
    Converts the timestamp column to datetime and sets it as the DataFrame index.
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(by=timestamp_col)
    df = df.set_index(timestamp_col)
    return df


def add_time_features(df: DataFrame) -> DataFrame:
    """
    Adds time-based features (Minute, Hour, DayOfYear) to the DataFrame.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index is not of type DatetimeIndex.")

    # df['Minute'] = df.index.minute
    df['Hour'] = df.index.hour
    df['DayOfYear'] = df.index.dayofyear
    return df


def fill_missing_values(df: DataFrame, target_col: str) -> DataFrame:
    """
    Fills missing values in the DataFrame.
    Target column NaNs are filled with 0, other columns are forward and backward filled.
    """
    df[target_col] = df[target_col].fillna(0)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df


def move_target_col(df: DataFrame, target_col: str) -> DataFrame:
    """
    Moves the target column to the end of the DataFrame.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    return df[feature_cols + [target_col]]
