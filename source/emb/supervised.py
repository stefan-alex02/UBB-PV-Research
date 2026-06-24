"""
Supervised forecasting probe on top of frozen embeddings.

Design
------
The CORE regressor is always trained on the EMBEDDING Z (the representation under
test) and predicts the target y. This is the "is this embedding useful for
forecasting?" probe, evaluated with time-series cross-validation.

For EXPLAINABILITY (feature importance + SHAP) we want attributions in terms of the
RAW, named input features — not opaque latent dimensions. We therefore wrap the
trained core model behind the (frozen) embedding transform:

        raw window  --embed_fn-->  embedding Z  --core_model-->  y_hat

and run permutation importance / SHAP on this wrapper using the raw windowed inputs.
`embed_fn` is the real, frozen transform (encoder for autoencoder; column selection
for identity / FS), so perturbed raw inputs produced by the explainer are mapped
through exactly the same pipeline the core model was trained on. This is what makes
raw-feature attribution possible even when the core model only ever sees embeddings.

Windowed-feature attributions are aggregated back to original columns by averaging
over the per-timestep copies of each feature.

Models: "xgboost", "random_forest", "mlp" (configurable params).
"""

import re
import numpy as np
from collections import defaultdict

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Model factory

def _make_model(model_name: str, params: dict, random_seed: int):
    """Construct a fresh regressor. MLP is wrapped with a scaler (NNs need scaled inputs)."""
    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=params["xgb_n_estimators"],
            max_depth=params["xgb_max_depth"],
            learning_rate=params["xgb_learning_rate"],
            subsample=params["xgb_subsample"],
            n_jobs=-1, random_state=random_seed, verbosity=0,
        )
    elif model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=params["rf_n_estimators"],
            max_depth=params["rf_max_depth"],
            n_jobs=-1, min_samples_leaf=1,
            max_features=1.0,
            random_state=random_seed,
        )
    elif model_name == "mlp":
        mlp = MLPRegressor(
            hidden_layer_sizes=tuple(params["mlp_hidden_layers"]),
            alpha=params["mlp_alpha"],
            learning_rate_init=params["mlp_learning_rate"],
            max_iter=params["mlp_max_iter"],
            early_stopping=True, random_state=random_seed,
        )
        return make_pipeline(StandardScaler(), mlp)
    else:
        raise ValueError(f"Unknown SUPERVISED_MODEL: {model_name!r}. "
                         "Choose 'xgboost', 'random_forest', or 'mlp'.")


def _is_tree(model_name: str) -> bool:
    return model_name in ("xgboost", "random_forest")

def _metrics(y_true, y_pred, y_range):
    """Regression metrics. NRMSE is RMSE normalized by the global y range
    (max - min) — the capacity-style normalization standard in PV forecasting."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "r2":    float(r2_score(y_true, y_pred)),
        "mae":   float(mean_absolute_error(y_true, y_pred)),
        "rmse":  rmse,
        "nrmse": float(rmse / y_range) if y_range > 0 else float("nan"),
    }


# Cross-validation (on embeddings)

def _cv_evaluate(Z, y, model_name, params, n_splits, random_seed):
    """TimeSeriesSplit CV on the embedding. Returns per-fold metrics and their means."""
    y_range = float(y.max() - y.min())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fold, (tr, te) in enumerate(tscv.split(Z), 1):
        model = _make_model(model_name, params, random_seed)
        model.fit(Z[tr], y[tr])
        pred = model.predict(Z[te])
        m = _metrics(y[te], pred, y_range)
        m.update({"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te))})
        fold_rows.append(m)
        print(f"    Fold {fold}/{n_splits}: train={len(tr):6d} test={len(te):6d}  "
              f"R²={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}")

    arr = lambda key: np.array([r[key] for r in fold_rows])
    summary = {}
    for key in ("r2", "mae", "rmse", "nrmse"):
        summary[f"{key}_mean"] = float(arr(key).mean())
        summary[f"{key}_std"]  = float(arr(key).std())
    print(f"    → R²={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}  "
          f"MAE={summary['mae_mean']:.4f}  RMSE={summary['rmse_mean']:.4f}  "
          f"NRMSE={summary['nrmse_mean']:.4f}±{summary['nrmse_std']:.4f}")
    return fold_rows, summary


# Importance aggregation

def _strip_timestep(name: str) -> str:
    """'GHI_t-3' → 'GHI'. Leaves names without the suffix unchanged."""
    return re.sub(r"_t-\d+$", "", name)


def _aggregate_to_columns(importances, feature_names):
    """Average importances over the per-timestep copies of each original column."""
    agg, cnt = defaultdict(float), defaultdict(int)
    for v, name in zip(importances, feature_names):
        base = _strip_timestep(name)
        agg[base] += float(v)
        cnt[base] += 1
    return {col: agg[col] / cnt[col] for col in agg}


# Permutation importance on the raw->embed->predict wrapper

def _permutation_importance_raw(predict_fn, X_raw, y, n_repeats, rng):
    """
    Model-agnostic permutation importance on RAW features, measured as the drop in
    R² when each raw column is shuffled. Works through the embedding wrapper, so the
    attributions are in raw-feature space regardless of the embedding method.
    """
    base = r2_score(y, predict_fn(X_raw))
    n_features = X_raw.shape[1]
    imps = np.zeros(n_features)
    X_work = X_raw.copy()
    for j in range(n_features):
        saved = X_work[:, j].copy()
        drops = np.empty(n_repeats)
        for r in range(n_repeats):
            X_work[:, j] = saved[rng.permutation(len(X_work))]
            drops[r] = base - r2_score(y, predict_fn(X_work))
        X_work[:, j] = saved          # restore column
        imps[j] = drops.mean()
    return imps


# SHAP on the wrapper (raw features)

def _shap_raw(predict_fn, X_raw, random_seed, max_samples, n_background=50):
    """
    KernelSHAP on the wrapper (model-agnostic, required because the wrapper is
    embed_fn ∘ core_model, not a bare tree). Returns mean |SHAP| per raw feature.

    WARNING: KernelSHAP is expensive — cost grows with raw feature count and the
    number of explained samples, and each evaluation runs the full embed→predict
    pipeline. Capped by max_samples; reduce it (or disable SHAP) if too slow.
    """
    import shap
    n = min(max_samples, len(X_raw))
    bg = shap.kmeans(X_raw, min(n_background, len(X_raw)))
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_raw[:n], silent=True)
    return np.abs(np.asarray(sv)).mean(axis=0), n


# Public entry point

def run_supervised_probe(
    Z: np.ndarray,
    X_windows: np.ndarray,
    y: np.ndarray,
    window_feature_names: list[str],
    embed_fn,
    model_name: str,
    model_params: dict,
    n_splits: int,
    random_seed: int,
    run_shap: bool = False,
    shap_max_samples: int = 200,
) -> dict:
    """
    CV the core model on embeddings; explain on raw features through the wrapper.

    Parameters
    ----------
    Z          : embeddings the core model trains/evaluates on, shape (N, D).
    X_windows  : raw windowed features for explainability, shape (N, F_win).
    embed_fn   : callable raw (N, F_win) -> embedding (N, D); the frozen transform
                 (encoder for AE, column-select for identity/FS, incl. any embedding
                 standardization) so wrapper(raw) reproduces Z.
    """
    print(f"\n{'='*70}")
    print(f"  Supervised probe — model={model_name!r}, CV folds={n_splits}")
    print(f"  Core model trained on EMBEDDING ({Z.shape[1]}-dim); "
          f"importance/SHAP on RAW ({X_windows.shape[1]}-dim).")
    print(f"{'='*70}")

    Z = np.asarray(Z, dtype=np.float32)
    X_windows = np.asarray(X_windows, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    rng = np.random.default_rng(random_seed)

    # ── 1. Cross-validation on embeddings ──
    folds, summary = _cv_evaluate(Z, y, model_name, model_params, n_splits, random_seed)

    # ── 2. Fit final core model on ALL embeddings (for explainability) ──
    print(f"  Fitting final core model on all {len(Z)} embeddings …")
    core_model = _make_model(model_name, model_params, random_seed)
    core_model.fit(Z, y)

    # ── 3. Build the raw→embed→predict wrapper and verify consistency ──
    def wrapper_predict(X_raw):
        return core_model.predict(embed_fn(np.asarray(X_raw, dtype=np.float32)))

    Z_check = embed_fn(X_windows)
    max_abs_diff = float(np.max(np.abs(np.asarray(Z_check, dtype=np.float32) - Z)))
    if max_abs_diff > 1e-3:
        print(f"  [Warning] embed_fn(X_windows) differs from provided Z "
              f"(max abs diff={max_abs_diff:.2e}). Raw attributions may be "
              f"inconsistent with the embedding the core model was trained on.")
    else:
        print(f"  [Check] embed_fn reproduces Z (max abs diff={max_abs_diff:.2e}). OK.")

    # ── 4. Feature names ──
    names = window_feature_names
    if names is None or len(names) != X_windows.shape[1]:
        names = [f"feat_{i}" for i in range(X_windows.shape[1])]

    result = {
        "model": model_name,
        "cv_folds": folds,
        "cv_summary": summary,
        "feature_names": names,
        "embed_consistency_max_abs_diff": max_abs_diff,
        "importance_kind": "permutation (raw, via wrapper)",
    }

    # ── 5. Permutation importance on raw ──
    print(f"  Computing permutation importance on raw features (through wrapper) …")
    perm = _permutation_importance_raw(wrapper_predict, X_windows, y, n_repeats=5, rng=rng)
    result["importances"] = perm.tolist()
    result["importances_by_column"] = _aggregate_to_columns(perm, names)

    # ── 6. SHAP on raw ──
    if run_shap:
        print(f"  Computing SHAP on raw features (through wrapper; KernelSHAP) …")
        print(f"    [Note] KernelSHAP cost scales with shap_max_samples (={shap_max_samples}) "
              f"× raw feature count; this can take minutes for the autoencoder. "
              f"Lower SHAP_MAX_SAMPLES or set RUN_SHAP=False if too slow.")
        try:
            mean_abs, n_used = _shap_raw(wrapper_predict, X_windows, random_seed, shap_max_samples)
            result["shap_mean_abs"] = np.asarray(mean_abs, dtype=float).tolist()
            result["shap_by_column"] = _aggregate_to_columns(mean_abs, names)
            print(f"    SHAP computed on {n_used} samples.")
        except Exception as e:
            print(f"    [Warning] SHAP failed ({type(e).__name__}: {e}). "
                  f"Permutation importance still available.")
            result["shap_mean_abs"] = None
            result["shap_by_column"] = None

    return result
