import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import SEED
from data_proc import load_raw_data, process_fold_data
from models import get_skorch_model


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    data_range = y_true.max() - y_true.min()
    nrmse = rmse / data_range if data_range != 0 else 0
    mask = y_true > 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else 0.0
    return {'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse, 'MAPE': mape, 'R2': r2}

def get_confidence_interval(data):
    if len(data) < 2: return np.mean(data), 0.0
    sem = stats.sem(data)
    h = sem * stats.t.ppf(0.975, len(data)-1)
    return np.mean(data), h


def run_benchmark(path, target_column='AC_POWER', model_name='LSTM', loading_method='sequence', seq_len=24, k_folds=5):
    """
    loading_method: 'standard', 'lagged', or 'sequence'
    """
    # 1. Load Raw Data (Features + Target together)
    raw_data, _ = load_raw_data(path, target_col=target_column)

    kf = KFold(n_splits=k_folds, shuffle=False)  # Shuffle False usually better for time-series blocks
    metrics_hist = {'MAE': [], 'RMSE': [], 'NRMSE': [], 'MAPE': [], 'R2': []}

    print(f"\n>>> Running {model_name} with '{loading_method}' data (Window={seq_len})...")

    # Raw Data Split
    for fold, (train_idx, test_idx) in enumerate(kf.split(raw_data)):

        # Split Raw Data first
        X_train_raw, X_test_raw = raw_data[train_idx], raw_data[test_idx]

        # 2. Process Data (Scale -> Lag/Sequence)
        # This handles all the complexity of reshaping
        X_train, y_train, X_test, y_test, scaler = process_fold_data(
            X_train_raw, X_test_raw, method=loading_method, seq_len=seq_len
        )

        # 3. Model Training
        model = None

        if model_name in ['ANN', 'LSTM']:
            model = get_skorch_model(model_name, X_train.shape)
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_test)

        elif model_name == 'XGBoost':
            # XGB needs 2D data. If 'sequence' passed by mistake, warn or flatten.
            if len(X_train.shape) > 2:
                # Flatten 3D to 2D: (N, T, F) -> (N, T*F)
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            # Internal Split for XGB Early Stopping
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

            model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=SEED)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            y_pred_scaled = model.predict(X_test).reshape(-1, 1)

        elif model_name in ['RF', 'SVM']:
            if len(X_train.shape) > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            if model_name == 'RF':
                model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
            else:
                model = SVR(kernel='rbf', C=10)

            model.fit(X_train, y_train.ravel())
            y_pred_scaled = model.predict(X_test).reshape(-1, 1)

        # 4. Inverse Transform Target (To get real kW error)
        # Note: process_fold_data scaled the whole dataset (including y).
        # We need to inverse transform Y using the scaler's last column stats.

        # Create a dummy matrix to inverse transform ONLY the target column
        # This is a bit hacky because scaler works on all columns
        # Shape of y is (N, 1). We need (N, Full_Cols).

        def inverse_y(y_s):
            n_samples = y_s.shape[0]
            n_cols = scaler.mean_.shape[0]
            dummy = np.zeros((n_samples, n_cols))
            dummy[:, -1] = y_s.flatten()  # Put pred in last column
            return scaler.inverse_transform(dummy)[:, -1:]

        y_true = inverse_y(y_test)
        y_pred = inverse_y(y_pred_scaled)

        # 5. Metrics
        y_pred = np.maximum(y_pred, 0)  # Physics check
        m = calculate_metrics(y_true, y_pred)
        for k, v in m.items(): metrics_hist[k].append(v)
        print(f"Fold {fold + 1}: RMSE={m['RMSE']:.4f}")

    # Final Report
    print(f"\n=== Final Results: {model_name} ({loading_method}) ===")
    for k in metrics_hist:
        mu, ci = get_confidence_interval(metrics_hist[k])
        print(f"{k:<10} | {mu:.4f} (+/- {ci:.4f})")

