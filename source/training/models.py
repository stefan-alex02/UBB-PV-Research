from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from xgboost import XGBRegressor

from source.training.data_process import ProcessingMethod
from source.training.modules import SampleLSTM, SampleANN

name_to_model: dict = {
    'ann': lambda **kwargs: SkorchModelWrapper(
        model=get_ann_model(**kwargs),
    ),
    'lstm': lambda **kwargs: SkorchModelWrapper(
        model=get_lstm_model(**kwargs),
    ),
    'xgboost': lambda val_size, seed, device, **kwargs: XGBoostModelWrapper(
        model=get_xgboost_model(seed=seed, device=device, **kwargs),
        val_size=val_size,
        seed=seed,
        device=device,
    ),
}


class ModelWrapper(ABC):
    def __init__(self):
        self.model: Any

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def get_loss_histories(self) -> tuple:
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SkorchModelWrapper(ModelWrapper):
    def __init__(self, model: NeuralNetRegressor):
        super().__init__()
        self.model: NeuralNetRegressor = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def get_loss_histories(self) -> tuple[Any, Any]:
        train_loss_history = self.model.history_[:, 'train_loss']
        valid_loss_history = self.model.history_[:, 'valid_loss']
        return train_loss_history, valid_loss_history

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModelWrapper(ModelWrapper):
    def __init__(self, model: XGBRegressor, val_size, seed, device):
        super().__init__()
        self.model: XGBRegressor = model
        self.val_size = val_size
        self.seed = seed
        self.device = device

    def train(self, X_train, y_train):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=self.val_size,
                                                    random_state=self.seed)
        self.model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=True)

    def get_loss_histories(self) -> tuple[Any, Any]:
        results = self.model.evals_result()
        train_loss_history = results['validation_0']['rmse'] # todo change to mse
        valid_loss_history = results['validation_1']['rmse']
        return train_loss_history, valid_loss_history

    def predict(self, X):
        return self.model.predict(X).reshape(-1, 1)


class StackedModelWrapper(ModelWrapper):
    def __init__(self, base_models: list[ModelWrapper], meta_model: ModelWrapper):
        super().__init__()
        self.base_models = base_models
        self.meta_model = meta_model

    def train(self, X_train, y_train):
        # Train base models
        base_model_predictions = []
        for model in self.base_models:
            model.train(X_train, y_train)
            preds = model.predict(X_train)
            base_model_predictions.append(preds)

        # Stack predictions for meta-model training
        stacked_X = torch.cat(base_model_predictions, dim=1)
        self.meta_model.train(stacked_X, y_train)

    def get_loss_histories(self) -> tuple[Any, Any]:
        # Combine loss histories from base models and meta model
        all_train_losses = []
        all_valid_losses = []

        for model in self.base_models:
            train_losses, valid_losses = model.get_loss_histories()
            all_train_losses.append(train_losses)
            all_valid_losses.append(valid_losses)

        meta_train_losses, meta_valid_losses = self.meta_model.get_loss_histories()
        all_train_losses.append(meta_train_losses)
        all_valid_losses.append(meta_valid_losses)

        return all_train_losses, all_valid_losses

    def predict(self, X):
        # Get predictions from base models
        base_model_predictions = []
        for model in self.base_models:
            preds = model.predict(X)
            base_model_predictions.append(preds)

        # Stack predictions for meta-model prediction
        stacked_X = torch.cat(base_model_predictions, dim=1)
        return self.meta_model.predict(stacked_X)


def get_early_stopping(patience):
    return EarlyStopping(monitor='valid_loss', patience=patience, threshold=0.0001, lower_is_better=True)


def get_xgboost_model(max_depth, n_estimators, lr, patience, seed, device, **_):
    return XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=lr,
        n_jobs=-1,
        early_stopping_rounds=patience,
        random_state=seed,
        # device=device # todo: enable support for gpu training
    )
    # todo suggestion: force non negative values (relu/softplus)


def get_lstm_model(processing_method: str, input_shape, epochs, batch_size, lr, module__hidden_units, module__num_layers,
                   module__dropout, module__bidirectional, patience, val_size, device, **_):
    early_stopping = get_early_stopping(patience)

    processing_method = processing_method if isinstance(processing_method, ProcessingMethod) else ProcessingMethod(processing_method)

    if processing_method == ProcessingMethod.WINDOW:
        n_features = input_shape[2]
    n_features = input_shape[2] # todo: might need to be changed based on processing method
    return NeuralNetRegressor(
        module=SampleLSTM,
        module__num_inputs=n_features,
        module__hidden_units=module__hidden_units,
        module__num_layers=module__num_layers,
        module__dropout=module__dropout,
        module__bidirectional=module__bidirectional,
        max_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        device=device,
        train_split=ValidSplit(cv=val_size),
        callbacks=[early_stopping],
        verbose=1
    )


def get_ann_model(input_shape, epochs, batch_size, lr, patience, val_size, device, **_):
    early_stopping = get_early_stopping(patience)

    return NeuralNetRegressor(
        module=SampleANN,
        module__num_inputs=input_shape[1],
        max_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        criterion=nn.MSELoss,
        device=device,
        train_split=ValidSplit(cv=val_size),
        callbacks=[early_stopping],
        verbose=1
    )
