import numpy as np
import pandas as pd


# DATA LOADING & PREPROC
def load_data(df):
    """
    Loads data and creates a dummy dataset if file not found for testing.
    """

    # Simple Preprocessing
    if 'DATE_TIME' in df.columns:
        df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df['Hour'] = df['DATE_TIME'].dt.hour
        df['Month'] = df['DATE_TIME'].dt.month
        df = df.drop(columns=['DATE_TIME'])

    # Handle NaNs
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Drop non-numeric IDs if present
    drop_cols = ['PLANT_ID', 'SOURCE_KEY']
    df = df.drop(columns=[c for c in df.columns if c in drop_cols], errors='ignore')

    target_col = 'AC_POWER' if 'AC_POWER' in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    return X, y
