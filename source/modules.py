import torch.nn as nn


class SampleANN(nn.Module):
    def __init__(self, num_inputs, num_units, dropout):
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
        # todo suggestion: force non negative values (relu/softplus)
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
        # todo suggestion: force non negative values (relu/softplus)
        return self.fc(last_step_out)
