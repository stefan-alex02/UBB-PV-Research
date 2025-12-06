from config import target_column, unisolar_site_key, dataset, k_folds, val_size, patience
from data_proc import load_raw_data, load_unisolar_data
from train_n_eval import run_benchmark, create_save_directory

# Dummy CSV for testing
# create_dummy_csv(path)

# Predefined configs
xgboost_config = {
    'model_name': 'XGBoost',
    'loading_method': 'lagged',
    'seq_len': 24,
    'n_estimators': 1000,
    'max_depth': 6,
    'lr': 0.05,
    'batch_size': 64,
}

xgboost_config_v1 = {
    'model_name': 'XGBoost',
    'loading_method': 'lagged',
    'seq_len': 96,
    'n_estimators': 1000,
    'max_depth': 6,
    'lr': 0.05,
    'batch_size': 64,
}

xgboost_config_st = {
    'model_name': 'XGBoost',
    'loading_method': 'standard',
    'n_estimators': 1000,
    'max_depth': 6,
    'lr': 0.05,
    'batch_size': 64,
}

lstm_config = {
    'model_name': 'LSTM',
    'loading_method': 'sequence',
    'seq_len': 24,
    'hidden_units': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
}

lstm_config_v1 = {
    'model_name': 'LSTM',
    'loading_method': 'sequence',
    'seq_len': 24,
    'hidden_units': 128,
    'num_layers': 3,
    'dropout': 0.3,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.0005,
}

lstm_config_v2 = {
    'model_name': 'LSTM',
    'loading_method': 'sequence',
    'seq_len': 24,
    'hidden_units': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
    'bidirectional': True,
}

lstm_config_v3 = {
    'model_name': 'LSTM',
    'loading_method': 'sequence',
    'seq_len': 48,
    'hidden_units': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
}

lstm_config_v4 = {
    'model_name': 'LSTM',
    'loading_method': 'sequence',
    'seq_len': 24,
    'hidden_units': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
}

ann_config = {
    'model_name': 'ANN',
    'loading_method': 'lagged',
    'seq_len': 24,
    'hidden_units': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 100,
    'lr': 0.001,
}

selected_config = ann_config

if dataset == 'private_au':
    df, dataset_name = load_raw_data("data/Solar_Energy_Generation_CK2.csv", target_col=target_column)
elif dataset == 'unisolar':
    df, dataset_name = load_unisolar_data("data/unisolar", target_site_key=unisolar_site_key)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

save_dir = create_save_directory(dataset_name, selected_config['model_name'])

run_benchmark(
    df, save_dir,
    model_name=selected_config['model_name'],
    loading_method=selected_config['loading_method'],
    seq_len=selected_config.get('seq_len', None),
    k_folds=k_folds,
    val_size=val_size,
    epochs=selected_config.get('epochs', None),
    max_depth=selected_config.get('max_depth', None),
    n_estimators=selected_config.get('n_estimators', None),
    batch_size=selected_config.get('batch_size', None),
    lr=selected_config.get('lr', None),
    patience=patience,
    hidden_units=selected_config.get('hidden_units', None),
    num_layers=selected_config.get('num_layers', None),
    dropout=selected_config.get('dropout', None)
)
