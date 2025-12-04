from config import target_column, model_name, loading_method, seq_len, k_folds, unisolar_site_key, dataset
from data_proc import load_raw_data, load_unisolar_data
from train_n_eval import run_benchmark, create_save_directory

# Dummy CSV for testing
# create_dummy_csv(path)

if dataset == 'private_au':
    df, dataset_name = load_raw_data("data/Solar_Energy_Generation_CK2.csv", target_col=target_column)
elif dataset == 'unisolar':
    df, dataset_name = load_unisolar_data("data/unisolar", target_site_key=unisolar_site_key)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

save_dir = create_save_directory(dataset_name, model_name)

run_benchmark(
    df, save_dir,
    model_name=model_name,
    loading_method=loading_method,
    seq_len=seq_len,
    k_folds=k_folds
)
