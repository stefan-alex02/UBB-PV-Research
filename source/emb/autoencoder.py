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
) -> Autoencoder:
    """Train an autoencoder on the windowed data X and return the trained model."""

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    input_dim = X.shape[1]
    model = Autoencoder(input_dim, latent_dim, hidden_dims, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    tensor_X = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # print(f"\n[Autoencoder] Architecture: {input_dim} -> {hidden_dims} -> {latent_dim} → {list(reversed(hidden_dims))} → {input_dim}")
    print(f"\n[Autoencoder] Architecture:")
    print(f"  Encoder: {input_dim} → {{{' → '.join(map(str, hidden_dims))}}} → {latent_dim}")
    print(f"  Decoder: {latent_dim} → {{{' → '.join(map(str, reversed(hidden_dims)))}}} → {input_dim}")
    print(f"[Autoencoder] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Autoencoder] Training for {epochs} epochs …")

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(X)
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  MSE loss = {epoch_loss:.6f}")

    print("[Autoencoder] Training complete.")
    return model


def get_autoencoder_embeddings(model: Autoencoder, X: np.ndarray, device: str) -> np.ndarray:
    """Encode X with a trained autoencoder and return numpy embeddings."""
    model.eval()
    tensor_X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model.encode(tensor_X)
    return Z.cpu().numpy()
