import os
from datetime import datetime

import numpy as np
import pandas as pd
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
    raw_data, _, dataset_name = load_raw_data(path, target_col=target_column)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create Path: [data_set]/[model_name]/[timestamp]
    save_dir = os.path.join(dataset_name, model_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Storage for detailed logs
    history_all_folds = []  # To store epoch-wise loss for every fold
    fold_results_list = []  # To store metrics for every fold

    kf = KFold(n_splits=k_folds, shuffle=False)
    metrics_hist = {'MAE': [], 'RMSE': [], 'NRMSE': [], 'MAPE': [], 'R2': []}

    print(f"\n>>> Running {model_name} with '{loading_method}' data (Window={seq_len})...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(raw_data)):

        X_train_raw, X_test_raw = raw_data[train_idx], raw_data[test_idx]

        X_train, y_train, X_test, y_test, scaler = process_fold_data(
            X_train_raw, X_test_raw, method=loading_method, seq_len=seq_len
        )

        model = None

        train_loss_history = []
        valid_loss_history = []

        if model_name in ['ANN', 'LSTM']:
            model = get_skorch_model(model_name, X_train.shape)
            model.fit(X_train, y_train)

            train_loss_history = model.history_[:, 'train_loss']
            valid_loss_history = model.history_[:, 'valid_loss']

            y_pred_scaled = model.predict(X_test)

        elif model_name == 'XGBoost':
            if len(X_train.shape) > 2: # XGB needs 2D data
                raise ValueError("invalid loading method for XGBoost")

            # Internal Split for XGB Early Stopping
            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

            model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=SEED)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            results = model.evals_result()
            train_loss_history = results['validation_0']['rmse']
            valid_loss_history = results['validation_1']['rmse']

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

        if len(train_loss_history) > 0:
            df_hist = pd.DataFrame({
                'Fold': fold + 1,
                'Epoch': range(1, len(train_loss_history) + 1),
                'Train_Loss': train_loss_history,
                'Valid_Loss': valid_loss_history
            })
            history_all_folds.append(df_hist)

        def inverse_y(y_s):
            n_samples = y_s.shape[0]
            n_cols = scaler.mean_.shape[0]
            dummy = np.zeros((n_samples, n_cols))
            dummy[:, -1] = y_s.flatten()  # Put pred in last column
            return scaler.inverse_transform(dummy)[:, -1:]

        y_true = inverse_y(y_test)
        y_pred = inverse_y(y_pred_scaled)

        # Metrics
        y_pred = np.maximum(y_pred, 0)  # Physics check
        metrics = calculate_metrics(y_true, y_pred)

        # Add Fold number to metrics for CSV saving
        metrics_with_fold = metrics.copy()
        metrics_with_fold['Fold'] = fold + 1
        fold_results_list.append(metrics_with_fold)

        # Keep existing list for Summary calculation
        for k, v in metrics.items(): metrics_hist[k].append(v)

        print(f"Fold {fold + 1}: RMSE={metrics['RMSE']:.4f}")

    print(f"\n=== Final Results: {model_name} ({loading_method}) ===")
    for k in metrics_hist:
        mu, ci = get_confidence_interval(metrics_hist[k])
        print(f"{k:<10} | {mu:.4f} (+/- {ci:.4f})")

