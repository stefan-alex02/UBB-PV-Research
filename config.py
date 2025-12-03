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

# Data set specific config
# path = "data/dummy.csv"
path = "data/Solar_Energy_Generation_CK2.csv"
target_column = 'SolarGeneration'

# Data loading config
loading_method = 'sequence'  # 'standard', 'lagged', 'sequence'
seq_len = 24  # For lagged and sequence methods

# Model config
model_name = 'LSTM'  # 'ANN', 'LSTM', 'RandomForest', 'SVR', 'XGBoost'
