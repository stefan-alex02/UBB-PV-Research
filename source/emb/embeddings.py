"""
All embedding / feature-extraction methods.

Methods
-----------------
identity         — Raw window vectors (no transformation).
autoencoder      — PyTorch autoencoder.
univariate_fs    — Univariate feature selection via F-statistic (sklearn).
random_forest_fs — Feature importance from a Random Forest regressor.
relieff          — ReliefF feature weighting (skrebate).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from skrebate import ReliefF

import config
from autoencoder import train_autoencoder, get_autoencoder_embeddings


def make_embed_fn(method: str, embed_meta: dict, scaler=None, device: str = "cpu"):
    """
    Build the frozen transform raw_window -> embedding, matching exactly how Z was
    produced (including the optional embedding standardization `scaler`).

    Used by the supervised probe so that explainability can run on RAW features:
    the explainer perturbs raw windows, which are mapped through this function to the
    embedding space the core model was trained on. This must reproduce the saved Z.

    method     : the embedding method name.
    embed_meta : the meta dict returned by get_embeddings (holds the trained AE model
                 or the selected feature indices).
    scaler     : a fitted StandardScaler applied to the embedding (or None).
    """
    def _maybe_scale(Z):
        return scaler.transform(Z) if scaler is not None else Z

    if method == "autoencoder":
        model = embed_meta["model"]
        def f(X):
            Z = get_autoencoder_embeddings(model, np.asarray(X, dtype=np.float32), device)
            return _maybe_scale(Z).astype(np.float32)
        return f

    elif method in ("univariate_fs", "random_forest_fs", "relieff"):
        idx = embed_meta["selected_indices"]
        def f(X):
            Z = np.asarray(X, dtype=np.float32)[:, idx]
            return _maybe_scale(Z).astype(np.float32)
        return f

    elif method == "identity":
        def f(X):
            Z = np.asarray(X, dtype=np.float32)
            return _maybe_scale(Z).astype(np.float32)
        return f

    else:
        raise ValueError(f"make_embed_fn: unknown method {method!r}.")


def get_embeddings(X_windows: np.ndarray, method: str, y: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """
    Compute embeddings from windowed vectors.

    Parameters
    ----------
    X_windows : np.ndarray  shape (N, k*F)  — windowed + concatenated rows
    method    : str         — one of the supported method names
    y         : np.ndarray  shape (N,)      — per-window target (used by FS methods).
                                              If None, FS methods fall back to X[:, 0].

    Returns
    -------
    embeddings : np.ndarray  shape (N, D)
    meta       : dict        — extra info (selected indices, model, …)
    """
    print(f"\n[Embedding] Method: {method!r}  |  Input dim: {X_windows.shape[1]}")

    if method == "identity":
        return _identity(X_windows)

    elif method == "autoencoder":
        return _autoencoder(X_windows)

    elif method == "univariate_fs":
        return _univariate_fs(X_windows, config.FS_N_FEATURES, y)

    elif method == "random_forest_fs":
        return _random_forest_fs(X_windows, config.FS_N_FEATURES, config.RF_N_ESTIMATORS, y)

    elif method == "relieff":
        return _relieff(X_windows, config.FS_N_FEATURES, config.RELIEFF_N_NEIGHBORS, y)

    else:
        raise ValueError(
            f"Unknown embedding method: {method!r}. "
            "Choose from: identity, autoencoder, univariate_fs, random_forest_fs, relieff."
        )


def _identity(X: np.ndarray) -> tuple[np.ndarray, dict]:
    print(f"[Embedding] Identity: keeping all {X.shape[1]} dimensions.")
    return X.copy(), {"method": "identity"}


def _autoencoder(X: np.ndarray) -> tuple[np.ndarray, dict]:
    model, history = train_autoencoder(
        X,
        latent_dim=config.AE_LATENT_DIM,
        hidden_dims=config.AE_HIDDEN_DIMS,
        epochs=config.AE_EPOCHS,
        batch_size=config.AE_BATCH_SIZE,
        lr=config.AE_LEARNING_RATE,
        dropout=config.AE_DROPOUT,
        device=config.DEVICE,
        random_seed=config.RANDOM_SEED,
        val_split=config.AE_VAL_SPLIT,
        weight_decay=config.AE_WEIGHT_DECAY,
        auto_n_layers=config.AE_AUTO_N_LAYERS,
    )
    Z = get_autoencoder_embeddings(model, X, config.DEVICE)
    print(f"[Embedding] Autoencoder output dim: {Z.shape[1]}")
    return Z, {"method": "autoencoder", "model": model, "history": history}


def _univariate_fs(X: np.ndarray, k: int, y: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """
    Univariate feature selection using the F-statistic between each feature
    and the first feature (used as a proxy target for an unsupervised setting).

    In practice, the first feature of the window is SolarGeneration at the
    latest time step — a natural regression target.
    """
    k = min(k, X.shape[1])
    y_proxy = y if y is not None else X[:, 0]

    selector = SelectKBest(f_regression, k=k)
    X_sel = selector.fit_transform(X, y_proxy)
    selected_indices = selector.get_support(indices=True).tolist()

    scores = selector.scores_
    print(f"[Embedding] Univariate FS: {X.shape[1]} -> {k} features selected.")
    print(f"            Top-5 feature indices by F-score: {np.argsort(scores)[::-1][:5].tolist()}")
    return X_sel.astype(np.float32), {
        "method": "univariate_fs",
        "selected_indices": selected_indices,
        "scores": scores,
    }


def _random_forest_fs(X: np.ndarray, k: int, n_estimators: int, y: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """
    Feature importance via Random Forest. Uses SolarGeneration (index 0) as target.
    """
    k = min(k, X.shape[1])
    y_proxy = y if y is not None else X[:, 0]

    print(f"[Embedding] Random Forest FS: fitting {n_estimators} trees …")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X, y_proxy)
    importances = rf.feature_importances_

    top_k = np.argsort(importances)[::-1][:k]
    X_sel = X[:, top_k]

    print(f"[Embedding] RF FS: {X.shape[1]} -> {k} features selected.")
    print(f"            Top-5 feature indices by importance: {top_k[:5].tolist()}")
    return X_sel.astype(np.float32), {
        "method": "random_forest_fs",
        "selected_indices": top_k.tolist(),
        "importances": importances,
    }


def _relieff(X: np.ndarray, k: int, n_neighbors: int, y: np.ndarray = None) -> tuple[np.ndarray, dict]:
    """
    ReliefF feature weighting (regression variant via skrebate).
    Uses SolarGeneration (index 0) as target.
    """
    k = min(k, X.shape[1])
    y_proxy = y if y is not None else X[:, 0]

    print(f"[Embedding] ReliefF: n_neighbors={n_neighbors}, selecting {k} features …")
    fs = ReliefF(n_features_to_select=k, n_neighbors=n_neighbors)
    X_sel = fs.fit_transform(X, y_proxy)
    feature_importances = fs.feature_importances_
    selected_indices = np.argsort(feature_importances)[::-1][:k].tolist()

    print(f"[Embedding] ReliefF: {X.shape[1]} -> {k} features selected.")
    return X_sel.astype(np.float32), {
        "method": "relieff",
        "selected_indices": selected_indices,
        "feature_importances": feature_importances,
    }