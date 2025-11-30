import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from config import device


class SolarModule(nn.Module):
    def __init__(self, num_inputs=10, num_units=64, dropout=0.1):
        super(SolarModule, self).__init__()
        self.dense0 = nn.Linear(num_inputs, num_units)
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(num_units, num_units // 2)
        self.output = nn.Linear(num_units // 2, 1)
        self.relu = nn.ReLU()

    def forward(self, X):
        X = self.relu(self.dense0(X))
        X = self.dropout(X)
        X = self.relu(self.dense1(X))
        X = self.output(X)
        return X


# --- B. Skorch Wrapper with Early Stopping ---
def get_ann_model(input_dim):
    """
    Returns a skorch NeuralNetRegressor with EarlyStopping enabled.
    """
    # EarlyStopping callback monitors 'valid_loss' by default
    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=10,
        threshold=0.0001,
        lower_is_better=True,
    )

    net = NeuralNetRegressor(
        module=SolarModule,
        module__num_inputs=input_dim,
        max_epochs=100,
        lr=0.001,
        batch_size=64,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        device=device,
        # train_split automatically creates the validation set for us
        # ValidSplit(0.1) reserves 10% of training data for validation
        callbacks=[early_stopping],
        verbose=1  # Set to 1 to see epoch logs
    )
    return net
