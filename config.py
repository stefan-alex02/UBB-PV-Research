import numpy as np

import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# General data set config
TIMESTAMP_NAMES = ['DATE_TIME', 'Timestamp']

##################################################################

# todo note: private_au is actually unisolar data for the site key 1 and with different collected features

# Data set specific config
dataset = "private_au" # 'private_au', 'unisolar'
target_column = 'SolarGeneration'
unisolar_site_key = 2

# Data loading config
loading_method = 'standard'  # 'standard', 'lagged', 'sequence'
seq_len = 24  # For lagged and sequence methods
val_size = 0.1 # For validation during training

# Model config
model_name = 'ANN'  # 'ANN', 'LSTM', 'RandomForest', 'SVR', 'XGBoost'
trees = 1000 # 100 for RF

# Training config
k_folds = 5
epochs = 10 # 100 for ANN
batch_size = 32
lr = 0.001 # 0.005 for LSTM, 0.05 for XGBoost
patience = 15

# Results config
RESULTS_DIR = "results"
