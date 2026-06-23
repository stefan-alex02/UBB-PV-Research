"""
Supervised forecasting probe on top of embeddings.

Purpose
-------
A *frozen* probe: take the already-computed representation and train a supervised
regressor to predict the target y, evaluated with time-series cross-validation.
This answers "does this representation contain forecast-relevant information?",
which complements the unsupervised clustering metrics.

Two inputs can be probed (configurable via SUPERVISED_ON):
  - "embedding" : probe on Z (the embedding). Score = embedding usefulness.
                  Dimensions are latent/opaque, so importances are not interpretable
                  for the autoencoder (they are for identity/FS where Z is raw columns).
  - "raw"       : probe on the raw windowed features X_windows. Always interpretable —
                  importances/SHAP map back to named original columns. Also a baseline
                  for how well one can forecast directly from raw windows.

Models (configurable): "xgboost", "random_forest", "mlp".

Time-series CV
--------------
Uses sklearn TimeSeriesSplit (expanding-window forward chaining): each fold trains on
the past and tests on the future, never shuffling, so no temporal leakage.

Explainability
--------------
- Tree models (xgboost, random_forest): exact, fast SHAP via TreeExplainer + the
  model's built-in feature_importances_.
- MLP: model-agnostic permutation importance (and optional SHAP, which is slow).
Windowed-feature importances are also aggregated back to original columns by averaging
over the per-timestep copies of each feature.
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
from sklearn.inspection import permutation_importance


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


# Cross-validation

def _cv_evaluate(Xin, y, model_name, params, n_splits, random_seed):
    """TimeSeriesSplit CV. Returns per-fold (r2, mae, rmse) and the means."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fold, (tr, te) in enumerate(tscv.split(Xin), 1):
        model = _make_model(model_name, params, random_seed)
        model.fit(Xin[tr], y[tr])
        pred = model.predict(Xin[te])
        r2   = r2_score(y[te], pred)
        mae  = mean_absolute_error(y[te], pred)
        rmse = float(np.sqrt(mean_squared_error(y[te], pred)))
        fold_rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                          "r2": float(r2), "mae": float(mae), "rmse": rmse})
        print(f"    Fold {fold}/{n_splits}: train={len(tr):6d} test={len(te):6d}  "
              f"R²={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    arr = lambda key: np.array([r[key] for r in fold_rows])
    summary = {
        "r2_mean": float(arr("r2").mean()),   "r2_std": float(arr("r2").std()),
        "mae_mean": float(arr("mae").mean()), "mae_std": float(arr("mae").std()),
        "rmse_mean": float(arr("rmse").mean()),"rmse_std": float(arr("rmse").std()),
    }
    print(f"    → R²={summary['r2_mean']:.4f}±{summary['r2_std']:.4f}  "
          f"MAE={summary['mae_mean']:.4f}  RMSE={summary['rmse_mean']:.4f}")
    return fold_rows, summary


# Importance & SHAP (fit final model on all data)

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


def _explain(Xin, y, model_name, params, feature_names, random_seed,
             run_shap, shap_max_samples):
    """Fit a final model on all data and compute importances (+ SHAP if enabled)."""
    model = _make_model(model_name, params, random_seed)
    model.fit(Xin, y)

    n_features = Xin.shape[1]
    # Feature names: if none/length mismatch (e.g. embedding dims), use generic labels.
    if feature_names is None or len(feature_names) != n_features:
        feature_names = [f"dim_{i}" for i in range(n_features)]

    result = {"feature_names": feature_names}

    # ── Built-in / permutation importance ──
    if _is_tree(model_name):
        importances = np.asarray(model.feature_importances_, dtype=float)
        result["importance_kind"] = "impurity/gain"
    else:
        pi = permutation_importance(model, Xin, y, n_repeats=5,
                                    random_state=random_seed, n_jobs=-1)
        importances = np.asarray(pi.importances_mean, dtype=float)
        result["importance_kind"] = "permutation"

    result["importances"] = importances.tolist()
    result["importances_by_column"] = _aggregate_to_columns(importances, feature_names)

    # ── SHAP (optional) ──
    if run_shap:
        try:
            import shap
            if _is_tree(model_name):
                explainer = shap.TreeExplainer(model)
                n = min(shap_max_samples, len(Xin))
                sv = explainer.shap_values(Xin[:n])
                mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
            else:
                # Model-agnostic, slower: use a small background + sample subset.
                n = min(shap_max_samples, len(Xin))
                bg = shap.sample(Xin, min(50, len(Xin)), random_state=random_seed)
                explainer = shap.KernelExplainer(model.predict, bg)
                sv = explainer.shap_values(Xin[:n], nsamples=100)
                mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
            result["shap_mean_abs"] = np.asarray(mean_abs, dtype=float).tolist()
            result["shap_by_column"] = _aggregate_to_columns(mean_abs, feature_names)
            print(f"    SHAP computed on {n} samples ({result['importance_kind']} model).")
        except Exception as e:
            print(f"    [Warning] SHAP failed ({type(e).__name__}: {e}). "
                  f"Skipping SHAP; permutation/impurity importance still available.")
            result["shap_mean_abs"] = None

    return result


# Public entry point

def run_supervised_probe(
    Z: np.ndarray,
    X_windows: np.ndarray,
    y: np.ndarray,
    window_feature_names: list[str],
    model_name: str,
    model_params: dict,
    n_splits: int,
    probe_on: list[str],
    random_seed: int,
    run_shap: bool = False,
    shap_max_samples: int = 500,
) -> dict:
    """
    Run the supervised forecasting probe.

    Returns
    -------
    dict keyed by input kind ("embedding" / "raw"), each with:
      cv_folds, cv_summary, importances, importances_by_column,
      shap_* (if enabled), feature_names.
    """
    print(f"\n{'='*70}")
    print(f"  Supervised probe — model={model_name!r}, CV folds={n_splits}, on={probe_on}")
    print(f"{'='*70}")

    inputs = {}
    if "embedding" in probe_on:
        inputs["embedding"] = (Z, None)  # latent dims: no interpretable names
    if "raw" in probe_on:
        inputs["raw"] = (X_windows, window_feature_names)

    results = {}
    for kind, (Xin, names) in inputs.items():
        print(f"\n  [{kind}] input shape: {Xin.shape}")
        Xin = np.asarray(Xin, dtype=np.float32)

        folds, summary = _cv_evaluate(Xin, y, model_name, model_params, n_splits, random_seed)

        print(f"  [{kind}] Fitting final model on all data for importance/SHAP …")
        expl = _explain(Xin, y, model_name, model_params, names, random_seed,
                        run_shap, shap_max_samples)

        results[kind] = {
            "model": model_name,
            "cv_folds": folds,
            "cv_summary": summary,
            **expl,
        }

    return results
