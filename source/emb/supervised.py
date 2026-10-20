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

Models: "xgboost", "random_forest", "mlp"
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, balanced_accuracy_score,
                             accuracy_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


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


def _make_classifier(model_name: str, params: dict, random_seed: int, n_classes: int):
    """Construct a fresh classifier that outputs class probabilities (softmax-style).
    MLP is wrapped with a scaler. Class weighting is applied at fit time via
    per-sample weights (uniform across model types), so no class_weight arg here."""
    if model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=params.get("xgb_n_estimators", 300),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_learning_rate", 0.1),
            subsample=params.get("xgb_subsample", 0.9),
            objective="multi:softprob", num_class=n_classes,
            n_jobs=-1, random_state=random_seed, verbosity=0,
            eval_metric="mlogloss",
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("rf_n_estimators", 300),
            max_depth=params.get("rf_max_depth", None),
            n_jobs=-1, random_state=random_seed,
        )
    elif model_name == "mlp":
        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(params.get("mlp_hidden_layers", [128, 64])),
            alpha=params.get("mlp_alpha", 1e-4),
            learning_rate_init=params.get("mlp_learning_rate", 1e-3),
            max_iter=params.get("mlp_max_iter", 300),
            early_stopping=True, random_state=random_seed,
        )
        return make_pipeline(StandardScaler(), mlp)
    else:
        raise ValueError(f"Unknown SUPERVISED_MODEL: {model_name!r}. "
                         "Choose 'xgboost', 'random_forest', or 'mlp'.")


def _fit_classifier(model, X, y, sample_weight=None):
    """Fit a classifier (or pipeline) with optional per-sample weights, handling the
    pipeline step-name prefix required by sklearn pipelines."""
    if sample_weight is None:
        model.fit(X, y)
    elif hasattr(model, "steps"):
        last = model.steps[-1][0]
        model.fit(X, y, **{f"{last}__sample_weight": sample_weight})
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


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


def _fit_capture_history(model_name, params, random_seed, Z_tr, y_tr, val_fraction):
    """
    Fit a fresh model on the training fold and, where the model supports it, capture
    a per iteration training curve (and a validation curve from a time ordered tail
    split of the training fold).

    The validation slice is the most recent `val_fraction` of the training fold (kept
    in chronological order, never shuffled) so it does not leak future information.

    Returns (fitted_model, history) where history is a dict with optional keys
    'iterations', 'train_curve', 'val_curve', or empty if the model has no curve.
    """
    history = {}

    # Build the internal time ordered validation tail
    n = len(Z_tr)
    use_val = val_fraction and val_fraction > 0.0 and n > 20
    if use_val:
        n_val = max(1, int(round(n * val_fraction)))
        tr_idx = np.arange(0, n - n_val)
        va_idx = np.arange(n - n_val, n)
    else:
        tr_idx = np.arange(0, n)
        va_idx = None

    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=params.get("xgb_n_estimators", 300),
            max_depth=params.get("xgb_max_depth", 6),
            learning_rate=params.get("xgb_learning_rate", 0.1),
            subsample=params.get("xgb_subsample", 0.9),
            n_jobs=-1, random_state=random_seed, verbosity=0,
            eval_metric="rmse",
        )
        if use_val:
            model.fit(Z_tr[tr_idx], y_tr[tr_idx],
                      eval_set=[(Z_tr[tr_idx], y_tr[tr_idx]), (Z_tr[va_idx], y_tr[va_idx])],
                      verbose=False)
            evals = model.evals_result()
            history = {
                "iterations": list(range(1, len(evals["validation_0"]["rmse"]) + 1)),
                "train_curve": [float(v) for v in evals["validation_0"]["rmse"]],
                "val_curve":   [float(v) for v in evals["validation_1"]["rmse"]],
                "curve_metric": "rmse",
            }

            model = _make_model(model_name, params, random_seed)
            model.fit(Z_tr, y_tr)
        else:
            model.fit(Z_tr, y_tr)

    elif model_name == "mlp":
        model = _make_model(model_name, params, random_seed)
        model.fit(Z_tr, y_tr)
        mlp = model.steps[-1][1] if hasattr(model, "steps") else model
        loss_curve = getattr(mlp, "loss_curve_", None)
        val_scores = getattr(mlp, "validation_scores_", None)
        if loss_curve is not None:
            history = {
                "iterations": list(range(1, len(loss_curve) + 1)),
                "train_curve": [float(v) for v in loss_curve],
                "curve_metric": "mlp_training_loss",
            }
            if val_scores is not None:
                history["val_curve"] = [float(v) for v in val_scores]
                history["val_curve_metric"] = "r2"

    else:  # random_forest or anything non iterative: no curve
        model = _make_model(model_name, params, random_seed)
        model.fit(Z_tr, y_tr)

    return model, history


def _ci95(values):
    """95% confidence interval half width for the mean (normal approx)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    return float(1.96 * v.std(ddof=1) / np.sqrt(n))


def _cv_evaluate(Z, y, model_name, params, n_splits, random_seed, val_fraction=0.0):
    """TimeSeriesSplit CV on the embedding. Returns per fold metrics, per fold training
    curves, and a summary including mean, std and 95% confidence intervals."""
    y_range = float(y.max() - y.min())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    fold_histories = []
    for fold, (tr, te) in enumerate(tscv.split(Z), 1):
        model, hist = _fit_capture_history(model_name, params, random_seed,
                                           Z[tr], y[tr], val_fraction)
        pred = model.predict(Z[te])
        m = _metrics(y[te], pred, y_range)
        m.update({"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te))})
        fold_rows.append(m)
        if hist:
            hist = dict(hist); hist["fold"] = fold
            fold_histories.append(hist)
        print(f"    Fold {fold}/{n_splits}: train={len(tr):6d} test={len(te):6d}  "
              f"R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}")

    arr = lambda key: np.array([r[key] for r in fold_rows])
    summary = {}
    for key in ("r2", "mae", "rmse", "nrmse"):
        vals = arr(key)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"]  = float(vals.std())
        summary[f"{key}_ci95"] = _ci95(vals)
    print(f"    => R2={summary['r2_mean']:.4f} +/- {summary['r2_ci95']:.4f} (95% CI)  "
          f"MAE={summary['mae_mean']:.4f}  RMSE={summary['rmse_mean']:.4f}  "
          f"NRMSE={summary['nrmse_mean']:.4f} +/- {summary['nrmse_ci95']:.4f}")
    return fold_rows, summary, fold_histories


def _per_class_regression(Z, y, y_classes, model_name, params, n_splits, random_seed):
    """
    For each CV fold, group the test samples by their y-class and compute regression
    metrics within each class; then average across folds. Lets you see how forecast
    accuracy differs between the night/low class and the higher-generation classes.

    Note: R2 within a near-constant class (e.g. the night/low class clustered near 0)
    can be unstable or negative because the within-class variance is tiny; MAE and
    NRMSE are more robust there. NRMSE here is normalized by the GLOBAL y range so the
    per-class numbers stay comparable to the overall NRMSE.
    """
    y_range = float(y.max() - y.min())
    classes = sorted(set(int(c) for c in y_classes))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    # collect per-fold, per-class metrics
    acc = {c: {"r2": [], "mae": [], "rmse": [], "nrmse": [], "n": []} for c in classes}

    for tr, te in tscv.split(Z):
        model = _make_model(model_name, params, random_seed)
        model.fit(Z[tr], y[tr])
        pred = model.predict(Z[te])
        yte_cls = y_classes[te]
        for c in classes:
            mask = yte_cls == c
            if mask.sum() < 2:
                continue
            m = _metrics(y[te][mask], pred[mask], y_range)
            for k in ("r2", "mae", "rmse", "nrmse"):
                acc[c][k].append(m[k])
            acc[c]["n"].append(int(mask.sum()))

    per_class = {}
    for c in classes:
        if not acc[c]["mae"]:
            continue
        per_class[c] = {
            "r2_mean":    float(np.mean(acc[c]["r2"])),
            "mae_mean":   float(np.mean(acc[c]["mae"])),
            "rmse_mean":  float(np.mean(acc[c]["rmse"])),
            "nrmse_mean": float(np.mean(acc[c]["nrmse"])),
            "avg_n_test": float(np.mean(acc[c]["n"])),
            "n_folds":    len(acc[c]["mae"]),
        }
    print("    Per-class regression (averaged across folds):")
    for c, m in per_class.items():
        print(f"      class {c}: R2={m['r2_mean']:.4f}  MAE={m['mae_mean']:.4f}  "
              f"NRMSE={m['nrmse_mean']:.4f}  (avg n_test={m['avg_n_test']:.0f})")
    return per_class


def _specificity_per_class(cm):
    """Per-class specificity (true negative rate) from a confusion matrix."""
    spec = {}
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec[i] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return spec


def _classification_metrics(y_true, y_pred, classes):
    """Macro and per-class metrics. Note: recall == sensitivity (true positive rate),
    so we report it once under 'recall' and also expose per-class specificity."""
    p_m, r_m, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0)
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    spec = _specificity_per_class(cm)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_m),
        "recall_macro": float(r_m),         # == sensitivity (macro)
        "f1_macro": float(f_m),
        "per_class": {
            int(c): {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),       # == sensitivity for this class
                "f1": float(f_c[i]),
                "specificity": float(spec.get(i, float("nan"))),
                "support": int(sup[i]),
            } for i, c in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }


def _cv_classify(Z, y_classes, model_name, params, n_splits, random_seed, weighted):
    """TimeSeriesSplit CV for classification. Returns per-fold metrics, a summary with
    means / std / 95% CI for the aggregate scores, and the summed confusion matrix."""
    classes = sorted(set(int(c) for c in y_classes))
    n_classes = len(classes)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for fold, (tr, te) in enumerate(tscv.split(Z), 1):
        clf = _make_classifier(model_name, params, random_seed, n_classes)
        sw = compute_sample_weight("balanced", y_classes[tr]) if weighted else None
        _fit_classifier(clf, Z[tr], y_classes[tr], sample_weight=sw)

        proba = clf.predict_proba(Z[te])
        pred = np.asarray(proba).argmax(axis=1)

        class_order = getattr(clf, "classes_", None)
        if class_order is None and hasattr(clf, "steps"):
            class_order = clf.steps[-1][1].classes_
        if class_order is not None:
            pred = np.asarray(class_order)[pred]
        m = _classification_metrics(y_classes[te], pred, classes)
        cm_total += np.array(m["confusion_matrix"])
        row = {"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)),
               "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"],
               "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
               "f1_macro": m["f1_macro"], "per_class": m["per_class"]}
        fold_rows.append(row)
        print(f"    Fold {fold}/{n_splits}: train={len(tr):6d} test={len(te):6d}  "
              f"bal_acc={m['balanced_accuracy']:.4f}  F1_macro={m['f1_macro']:.4f}  "
              f"P_macro={m['precision_macro']:.4f}  R_macro={m['recall_macro']:.4f}")

    arr = lambda key: np.array([r[key] for r in fold_rows])
    summary = {}
    for key in ("balanced_accuracy", "accuracy", "precision_macro", "recall_macro", "f1_macro"):
        vals = arr(key)
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"]  = float(vals.std())
        summary[f"{key}_ci95"] = _ci95(vals)

    per_class_avg = {}
    for c in classes:
        per_class_avg[int(c)] = {}
        for metric in ("precision", "recall", "f1", "specificity"):
            vals = [r["per_class"][c][metric] for r in fold_rows if c in r["per_class"]]
            per_class_avg[int(c)][metric] = float(np.nanmean(vals)) if vals else float("nan")
    summary["per_class_avg"] = per_class_avg
    summary["confusion_matrix_total"] = cm_total.tolist()
    summary["classes"] = classes

    print(f"    => balanced_acc={summary['balanced_accuracy_mean']:.4f} "
          f"+/- {summary['balanced_accuracy_ci95']:.4f} (95% CI)  "
          f"F1_macro={summary['f1_macro_mean']:.4f}  "
          f"P_macro={summary['precision_macro_mean']:.4f}  "
          f"R_macro={summary['recall_macro_mean']:.4f}")
    return fold_rows, summary

def _strip_timestep(name: str) -> str:
    """'GHI_t-3' -> 'GHI'. Leaves names without the suffix unchanged."""
    return re.sub(r"_t-\d+$", "", name)


def _aggregate_to_columns(importances, feature_names):
    """Average importances over the per-timestep copies of each original column."""
    agg, cnt = defaultdict(float), defaultdict(int)
    for v, name in zip(importances, feature_names):
        base = _strip_timestep(name)
        agg[base] += float(v)
        cnt[base] += 1
    return {col: agg[col] / cnt[col] for col in agg}


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


def _shap_raw(predict_fn, X_raw, random_seed, max_samples, n_background=50):
    """
    KernelSHAP on the wrapper (model-agnostic, required because the wrapper is
    embed_fn ∘ core_model, not a bare tree). Returns mean |SHAP| per raw feature.
    """
    import shap
    n = min(max_samples, len(X_raw))
    bg = shap.kmeans(X_raw, min(n_background, len(X_raw)))
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_raw[:n], silent=True)
    return np.abs(np.asarray(sv)).mean(axis=0), n


def _top_n(by_dict, n):
    """Return the top n (name, value) pairs from a dict, descending by value."""
    items = sorted(by_dict.items(), key=lambda kv: kv[1], reverse=True)
    return [{"name": k, "value": float(v)} for k, v in items[:n]]


def run_supervised_probe(
    Z: np.ndarray,
    X_windows: np.ndarray,
    y: np.ndarray,
    y_classes: np.ndarray,
    window_feature_names: list[str],
    embed_fn,
    model_name: str,
    model_params: dict,
    n_splits: int,
    random_seed: int,
    task: str = "regression",
    class_weighted: bool = False,
    run_shap: bool = False,
    shap_max_samples: int = 200,
    cv_val_fraction: float = 0.0,
    top_n: int = 15,
) -> dict:
    """
    Probe the embedding with a supervised model under time-series CV.

    task : "regression", "classification", or "both".
      - regression     : core model predicts continuous y; also reports a per-class
                          regression breakdown (metrics within each y-class).
      - classification : core model predicts the discretized y-class, outputting class
                          probabilities; reports precision/recall(=sensitivity)/F1,
                          balanced accuracy, per-class specificity and a confusion matrix.
    class_weighted : if True, classification uses balanced per-sample weights to counter
                     the dominant (night/low) class.

    Explainability (permutation importance + optional SHAP on RAW features through the
    embedding wrapper) is computed once, using the regression wrapper when regression is
    active, otherwise the classification wrapper (probability of the predicted class).

    Parameters mirror the earlier version; Z is what the core model trains on, X_windows
    and embed_fn enable raw-feature attribution.
    """
    print(f"\n{'='*70}")
    print(f"  Supervised probe: model={model_name!r}, task={task!r}, CV folds={n_splits}")
    print(f"  Core model trained on EMBEDDING ({Z.shape[1]}-dim); "
          f"importance/SHAP on RAW ({X_windows.shape[1]}-dim).")
    print(f"{'='*70}")

    Z = np.asarray(Z, dtype=np.float32)
    X_windows = np.asarray(X_windows, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    y_classes = np.asarray(y_classes).astype(int)
    rng = np.random.default_rng(random_seed)

    do_reg = task == "regression"
    do_clf = task == "classification"
    if not (do_reg or do_clf):
        raise ValueError(f"Unknown task {task!r}. Choose 'regression' or 'classification'.")

    result = {
        "model": model_name,
        "task": task,
        "class_weighted": bool(class_weighted),
        "top_n": top_n,
    }

    # Feature names
    names = window_feature_names
    if names is None or len(names) != X_windows.shape[1]:
        names = [f"feat_{i}" for i in range(X_windows.shape[1])]
    result["feature_names"] = names

    # Regression task
    if do_reg:
        print("\n  [Regression] Cross-validation on embeddings:")
        folds, summary, fold_histories = _cv_evaluate(
            Z, y, model_name, model_params, n_splits, random_seed, cv_val_fraction)
        per_class_reg = _per_class_regression(
            Z, y, y_classes, model_name, model_params, n_splits, random_seed)
        result["regression"] = {
            "cv_folds": folds,
            "cv_summary": summary,
            "cv_fold_histories": fold_histories,
            "cv_val_fraction": cv_val_fraction,
            "per_class": per_class_reg,
        }

    # Classification task
    if do_clf:
        print("\n  [Classification] Cross-validation on embeddings"
              f"{' (balanced class weights)' if class_weighted else ''}:")
        cfolds, csummary = _cv_classify(
            Z, y_classes, model_name, model_params, n_splits, random_seed, class_weighted)
        result["classification"] = {
            "cv_folds": cfolds,
            "cv_summary": csummary,
        }

    # Fit final core model on ALL embeddings
    print(f"\n  Fitting final core model on all {len(Z)} embeddings (for explainability) ...")
    if do_reg:
        core_model = _make_model(model_name, model_params, random_seed)
        core_model.fit(Z, y)

        def wrapper_predict(X_raw):
            return core_model.predict(embed_fn(np.asarray(X_raw, dtype=np.float32)))

        expl_target = y
        expl_kind = "regression output"
    else:
        n_classes = len(sorted(set(y_classes)))
        core_model = _make_classifier(model_name, model_params, random_seed, n_classes)
        sw = compute_sample_weight("balanced", y_classes) if class_weighted else None
        _fit_classifier(core_model, Z, y_classes, sample_weight=sw)

        def wrapper_predict(X_raw):
            proba = core_model.predict_proba(embed_fn(np.asarray(X_raw, dtype=np.float32)))
            return proba.max(axis=1)

        expl_target = wrapper_predict(X_windows)  # self-reference baseline for perm importance
        expl_kind = "classification max-class probability"

    # Consistency check
    Z_check = embed_fn(X_windows)
    max_abs_diff = float(np.max(np.abs(np.asarray(Z_check, dtype=np.float32) - Z)))
    if max_abs_diff > 1e-3:
        print(f"  [Warning] embed_fn(X_windows) differs from provided Z "
              f"(max abs diff={max_abs_diff:.2e}). Raw attributions may be inconsistent.")
    else:
        print(f"  [Check] embed_fn reproduces Z (max abs diff={max_abs_diff:.2e}). OK.")
    result["embed_consistency_max_abs_diff"] = max_abs_diff
    result["explain_target_kind"] = expl_kind

    # Permutation importance on raw
    print(f"  Computing permutation importance on raw features (through wrapper; "
          f"target = {expl_kind}) ...")
    perm = _permutation_importance_raw(wrapper_predict, X_windows, expl_target, n_repeats=5, rng=rng)
    perm_by_col = _aggregate_to_columns(perm, names)
    result["importance_kind"] = "permutation (raw, via wrapper)"
    result["importances"] = perm.tolist()
    result["importances_by_column"] = perm_by_col
    result["top_importance_features"] = _top_n(
        {names[i]: float(perm[i]) for i in range(len(names))}, top_n)
    result["top_importance_columns"] = _top_n(perm_by_col, top_n)

    # 6. SHAP on raw
    if run_shap:
        print(f"  Computing SHAP on raw features (through wrapper; KernelSHAP) ...")
        print(f"    [Note] KernelSHAP cost scales with shap_max_samples (={shap_max_samples}) "
              f"x raw feature count; this can take minutes for the autoencoder. "
              f"Lower SHAP_MAX_SAMPLES or set RUN_SHAP=False if too slow.")
        try:
            mean_abs, n_used = _shap_raw(wrapper_predict, X_windows, random_seed, shap_max_samples)
            shap_by_col = _aggregate_to_columns(mean_abs, names)
            result["shap_mean_abs"] = np.asarray(mean_abs, dtype=float).tolist()
            result["shap_by_column"] = shap_by_col
            result["top_shap_features"] = _top_n(
                {names[i]: float(mean_abs[i]) for i in range(len(names))}, top_n)
            result["top_shap_columns"] = _top_n(shap_by_col, top_n)
            print(f"    SHAP computed on {n_used} samples.")
        except Exception as e:
            print(f"    [Warning] SHAP failed ({type(e).__name__}: {e}). "
                  f"Permutation importance still available.")
            result["shap_mean_abs"] = None
            result["shap_by_column"] = None

    return result
