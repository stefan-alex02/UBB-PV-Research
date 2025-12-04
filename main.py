from config import path, target_column, model_name, loading_method, seq_len
from train_n_eval import run_benchmark

# Dummy CSV for testing
# create_dummy_csv(path)

run_benchmark(path, target_column=target_column, model_name=model_name, loading_method=loading_method, seq_len=seq_len)
