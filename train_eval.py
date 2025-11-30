import numpy as np
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from config import SEED
from data_load import load_data
from models import get_ann_model


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    data_range = y_true.max() - y_true.min()
    nrmse = rmse / data_range if data_range != 0 else 0

    # Avoid division by zero for MAPE
    mask = y_true > 1.0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    return {'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse, 'MAPE': mape, 'R2': r2}


def get_confidence_interval(data):
    """Computes mean and 95% CI margin"""
    if len(data) < 2: return np.mean(data), 0.0
    mean = np.mean(data)
    sem = stats.sem(data)
    # t-statistic for 95% CI
    h = sem * stats.t.ppf(0.975, len(data)-1)
    return mean, h


def run(df, model_name='ANN', k_folds=5):
    X, y = load_data(df)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_metrics = {'MAE': [], 'RMSE': [], 'NRMSE': [], 'MAPE': [], 'R2': []}

    print(f"\n>>> Starting {k_folds}-Fold CV for {model_name}...")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # 1. Outer CV Split
        X_train_full, X_test = X[train_idx], X[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # 2. Scaling (Fit on Train, Transform Test)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()  # Optional: Scale target for ANN/SVM helps convergence

        X_train_full = scaler_X.fit_transform(X_train_full)
        X_test = scaler_X.transform(X_test)

        # We scale Y for ANN/SVM, usually not needed for Trees
        y_train_scaled = scaler_y.fit_transform(y_train_full)

        # 3. Model Training Logic
        model = None

        if model_name == 'ANN':
            # skorch handles internal validation split automatically via 'train_split'
            model = get_ann_model(input_dim=X.shape[1])
            model.fit(X_train_full, y_train_scaled)

            # Predict
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        elif model_name == 'XGBoost':
            # XGBoost needs explicit internal split for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.1, random_state=SEED
            )

            model = XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=SEED,
                early_stopping_rounds=50
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            # Predict (Target scaling usually not needed for XGB, but consistent here)
            # If we didn't scale Y for XGB, remove inverse_transform
            # For this script consistency, we assume Y was scaled for all.
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        elif model_name in ['RF', 'SVM']:
            # No Early Stopping -> Use ALL training data
            if model_name == 'RF':
                model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
            else:
                model = SVR(kernel='rbf', C=10.0, epsilon=0.1)

            model.fit(X_train_full, y_train_scaled.ravel())
            y_pred_scaled = model.predict(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        # 4. Evaluation
        # Enforce physical constraint (Solar Power >= 0)
        y_pred = np.maximum(y_pred, 0)

        metrics = calculate_metrics(y_test, y_pred)
        for k, v in metrics.items():
            fold_metrics[k].append(v)

        print(f"Fold {fold + 1}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # 6. Final Report
    print(f"\n=== Final Results: {model_name} ===")
    print(f"{'Metric':<10} | {'Mean':<10} | {'95% CI (+/-)':<10}")
    print("-" * 35)
    for m in fold_metrics:
        mean_val, ci = get_confidence_interval(fold_metrics[m])
        print(f"{m:<10} | {mean_val:<10.4f} | {ci:<10.4f}")
