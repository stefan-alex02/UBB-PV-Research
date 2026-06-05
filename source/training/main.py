import os

from source.training.data_loader import drop_columns, make_timestamp_index, add_time_features, \
    fill_missing_values, move_target_col, loading_methods, get_num_features
from source.training.data_process import split_and_process_folds
from source.training.evaluation import process_fold_results, Results, save_results
from source.training.models import ModelWrapper, name_to_model
from source.training.util import load_config, setup_device, setup_seed, create_directory, logger, add_file_handler_to_logger, \
    save_config

device = setup_device()

config = load_config("config.yaml")

seed = config['general']['seed']
results_dir: str = config['general']['results_dir']
dataset_type: str = config['general']['selected_dataset_type']
model_name: str = config['general']['selected_model']

dataset_cfg = config['dataset'][dataset_type]
methodology_cfg = config['methodology']
model_cfg = config['model'][model_name]

setup_seed(seed)
save_dir = create_directory(results_dir, dataset_cfg['name'], model_name, timestamped=True)
add_file_handler_to_logger(save_dir)
save_config({'general': config['general'], 'dataset': dataset_cfg, 'methodology': methodology_cfg, 'model': model_cfg},
                os.path.join(save_dir, 'used_config.yaml'))

df = loading_methods[dataset_type](parent_dir=results_dir, **dataset_cfg)
df = make_timestamp_index(df, dataset_cfg['timestamp_col'])
df = add_time_features(df)
df = fill_missing_values(df, dataset_cfg['target_col'])
df = drop_columns(df, dataset_cfg['cols_to_drop'])
df = move_target_col(df, dataset_cfg['target_col'])
num_features = get_num_features(df)

results: Results = Results()

for (fold, X_train, y_train, X_test, y_test, scaler_y) in split_and_process_folds(
        df,
        k_folds=methodology_cfg['k_folds'],
        method=methodology_cfg['processing_method'],
        seq_len=methodology_cfg['seq_len']):

    model: ModelWrapper = name_to_model[model_name](**model_cfg, **methodology_cfg, seed=seed, device=device)

    model.train(X_train, y_train)
    train_loss_history, valid_loss_history = model.get_loss_histories()

    y_pred_scaled = model.predict(X_test)

    y_true = scaler_y.inverse_transform(y_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

    process_fold_results(results, fold, train_loss_history, valid_loss_history, y_true, y_pred)

    logger.info(results.get_fold_results_str_for(fold))

save_results(results, save_dir)
