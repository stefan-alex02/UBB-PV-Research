from util import target_column, unisolar_site_key, dataset, k_folds, val_size, patience
from train_n_eval import run, create_directory

selected_config = ann_config

if dataset == 'private_au':
    df, dataset_name = load_raw_data("data/Solar_Energy_Generation_CK2.csv", target_col=target_column)
elif dataset == 'unisolar':
    df, dataset_name = load_unisolar_data("data/unisolar", target_site_key=unisolar_site_key)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

save_dir = create_directory(dataset_name, selected_config['model_name'])

run(
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
