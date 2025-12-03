from config import path, target_column, model_name, loading_method, seq_len
from train_n_eval import run_benchmark

# Dummy CSV for testing
# create_dummy_csv(path)

# LSTM (Sequence Based - 3D)
run_benchmark(path, target_column=target_column, model_name=model_name, loading_method=loading_method, seq_len=seq_len)

# XGBoost (Lagged Based - 2D)
# run_benchmark(csv_path, model_name='XGBoost', loading_method='lagged', seq_len=24)
