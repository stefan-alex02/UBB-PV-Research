import torch
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping

from config import device, patience


class SampleANN(nn.Module):
    def __init__(self, num_inputs=10, num_units=64, dropout=0.1):
        super(SampleANN, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(num_inputs, num_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_units, num_units // 2),
            nn.ReLU(),
            nn.Linear(num_units // 2, 1)
        )

    def forward(self, X):
        return self.dense(X)


class SampleLSTM(nn.Module):
    def __init__(self, num_inputs, hidden_units=50, num_layers=2, dropout=0.2, bidirectional=False):
        super(SampleLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.lstm = nn.LSTM(num_inputs, hidden_units, num_layers,
                            batch_first=True, # batch_first=True expects input shape (Batch, Seq_Len, Features)
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_units if not bidirectional else hidden_units * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)


def get_skorch_model(model_type, input_shape, epochs, batch_size, lr, hidden_units=64, num_layers=2, dropout=0.2, bidirectional=False):
    """Factory for Skorch Wrappers"""
    early_stopping = EarlyStopping(monitor='valid_loss', patience=patience, threshold=0.0001, lower_is_better=True)

    if model_type == 'LSTM':
        n_features = input_shape[2]
        return NeuralNetRegressor(
            module=SampleLSTM,
            module__num_inputs=n_features,
            module__hidden_units=hidden_units,
            module__num_layers=num_layers,
            module__dropout=dropout,
            module__bidirectional=bidirectional,
            max_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer=torch.optim.Adam,
            criterion=nn.MSELoss,
            device=device,
            callbacks=[early_stopping],
            verbose=1
        )

    elif model_type == 'ANN':
        return NeuralNetRegressor(
            module=SampleANN,
            module__num_inputs=input_shape[1],
            max_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            optimizer=torch.optim.Adam,
            criterion=nn.MSELoss,
            device=device,
            callbacks=[early_stopping],
            verbose=1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
