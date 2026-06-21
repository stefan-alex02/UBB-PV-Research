"""
PyTorch Autoencoder for unsupervised embedding generation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _build_mlp(dims: list[int], dropout: float, activation=nn.ReLU) -> nn.Sequential:
    """Build a sequential MLP from a list of layer dimensions."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # No activation/dropout after last linear
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        """
        Symmetric autoencoder.

        Encoder: input_dim → hidden_dims → latent_dim
        Decoder: latent_dim → hidden_dims[::-1] → input_dim
        """
        super().__init__()
        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + list(reversed(hidden_dims)) + [input_dim]

        self.encoder = _build_mlp(encoder_dims, dropout)
        self.decoder = _build_mlp(decoder_dims, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X: np.ndarray,
    latent_dim: int,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    device: str,
    random_seed: int,
    val_split: float = 0.0,
    weight_decay: float = 0.0,
) -> tuple[Autoencoder, dict]:
    """
    Train an autoencoder on windowed data X.

    Parameters
    ----------
    val_split    : fraction of X held out for validation (0 disables validation).
    weight_decay : L2 regularization strength passed to Adam.

    Returns
    -------
    model   : trained Autoencoder
    history : dict with keys 'epoch', 'train_loss', 'val_loss' (val_loss empty if no split)
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    input_dim = X.shape[1]
    model = Autoencoder(input_dim, latent_dim, hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Train / validation split
    tensor_X = torch.tensor(X, dtype=torch.float32)
    full_ds = TensorDataset(tensor_X)

    if val_split and val_split > 0.0:
        n_val = int(len(full_ds) * val_split)
        n_train = len(full_ds) - n_val
        gen = torch.Generator().manual_seed(random_seed)
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=gen)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        train_ds, val_ds, val_loader = full_ds, None, None
        n_train, n_val = len(full_ds), 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    print(f"\n[Autoencoder] Architecture: {input_dim} → {hidden_dims} → {latent_dim} → {list(reversed(hidden_dims))} → {input_dim}")
    print(f"[Autoencoder] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Autoencoder] Train samples: {n_train}  |  Val samples: {n_val}  "
          f"|  weight_decay (L2): {weight_decay}")
    print(f"[Autoencoder] Training for {epochs} epochs …")

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= n_train

        # Validate
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            vloss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    recon, _ = model(batch)
                    vloss += criterion(recon, batch).item() * len(batch)
            val_loss = vloss / n_val

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # if epoch % max(1, epochs // 20) == 0 or epoch == 1:
        if val_loader is not None:
            print(f"  Epoch {epoch:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")
        else:
            print(f"  Epoch {epoch:4d}/{epochs}  train={train_loss:.6f}")

    print("[Autoencoder] Training complete.")
    return model, history


def get_autoencoder_embeddings(model: Autoencoder, X: np.ndarray, device: str) -> np.ndarray:
    """Encode X with a trained autoencoder and return numpy embeddings."""
    model.eval()
    tensor_X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(tensor_X)
    return Z.cpu().numpy()