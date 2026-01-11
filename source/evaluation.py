import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from source.util import save_as_csv, save_as_parquet, logger


class Results:
    def __init__(self):
        self.metrics_hist = {'MAE': [], 'RMSE': [], 'NRMSE': [], 'MAPE': [], 'R2': []}
        self.history_all_folds = []
        self.fold_results_list = []
        self.predictions_all_folds = []
        self.metrics_summary_df = None

    def get_fold_results_str_for(self, fold: int) -> str:
        for res in self.fold_results_list:
            if res['Fold'] == fold + 1:
                return ' - '.join([f"{k}: {v:.4f}" for k, v in res.items()])
        return f"Fold {fold + 1}: results not found."

    def get_metrics_summary_lines(self) -> list:
        lines = []
        for _, row in self.metrics_summary_df.iterrows():
            lines.append(f"{row['Metric']}: {row['Mean_Display']}")
        return lines


def process_fold_results(results: Results, fold: int, train_loss_history, valid_loss_history, y_true, y_pred):
    if len(train_loss_history) > 0 and len(valid_loss_history) > 0:
        results.history_all_folds.append(create_history_df(train_loss_history, valid_loss_history, fold))

    results.predictions_all_folds.append(create_fold_y_df(fold, y_true, y_pred))

    y_pred = np.maximum(y_pred, 0)  # todo: SOME OF THE VALUES ARE CORRECTED BY THIS
    metrics = calculate_metrics(y_true, y_pred)
    results.fold_results_list.append({'Fold': fold + 1, **metrics})
    for k, v in metrics.items(): results.metrics_hist[k].append(v)


def save_results(results: Results, save_dir: str):
    df_folds = pd.DataFrame(results.fold_results_list)
    save_as_csv(df_folds, save_dir, 'fold_metrics.csv')

    if results.history_all_folds:
        df_history = pd.concat(results.history_all_folds, ignore_index=True)
        save_as_csv(df_history, save_dir, 'training_losses.csv')

    if results.predictions_all_folds:
        df_preds = pd.concat(results.predictions_all_folds, ignore_index=True)
        save_as_parquet(df_preds, save_dir, 'predictions.parquet')

    results.metrics_summary_df = create_metrics_summary_df(results.metrics_hist)
    save_as_csv(results.metrics_summary_df, save_dir, 'summary_metrics.csv')

    logger.info("=== Summary Metrics ===")
    for line in results.get_metrics_summary_lines(): logger.info(line)

    logger.info(f"All files saved to: {save_dir}")


def calculate_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    data_range = y_true.max() - y_true.min()
    nrmse = rmse / data_range if data_range != 0 else 0
    mask = y_true > 1.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else 0.0
    return {'MAE': mae, 'RMSE': rmse, 'NRMSE': nrmse, 'MAPE': mape, 'R2': r2}


def create_history_df(train_loss_history, valid_loss_history, fold) -> pd.DataFrame:
    epochs = list(range(1, len(train_loss_history) + 1))
    df_hist = pd.DataFrame({
        'Fold': [fold + 1] * len(epochs),
        'Epoch': epochs,
        'Train_Loss': train_loss_history,
        'Valid_Loss': valid_loss_history
    })
    return df_hist


def create_fold_y_df(fold, y_true, y_pred) -> pd.DataFrame:
    return pd.DataFrame({
        'Fold': fold + 1,
        'Actual': y_true.flatten(),
        'Predicted': y_pred.flatten()
    })


def create_metrics_summary_df(hist) -> pd.DataFrame:
    summary_data = []
    for k in hist:
        mu, ci = get_confidence_interval(hist[k])
        summary_data.append({
            'Metric': k,
            'Mean': mu,
            '95% CI': ci,
            'Mean_Display': f"{mu:.4f} ± {ci:.4f}"
        })
    return pd.DataFrame(summary_data)


def get_confidence_interval(data):
    if len(data) < 2: return np.mean(data), 0.0
    sem = stats.sem(data)
    h = sem * stats.t.ppf(0.975, len(data)-1)
    return np.mean(data), h
