import os
import sys
import time
import numpy as np

# Make project root importable
sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader  import load_and_preprocess, build_windows, discretize_target
from embeddings   import get_embeddings
from clustering   import run_clustering
from analysis     import (
    pairwise_correlation_analysis,
    external_correlation_analysis,
    homogeneity_analysis,
    cluster_summary,
    compare_clusterings,
    save_results,
)
from supervised   import run_supervised_probe
from plot_results import plot


def main():
    t0 = time.time()
    rng = np.random.default_rng(config.RANDOM_SEED)

    print("=" * 70)
    print("  PV Energy Forecasting: Unsupervised Embedding Pipeline")
    print("=" * 70)

    # 1. Load & normalise
    X, y_raw, feature_names, target_idx = load_and_preprocess(
        path=config.DATASET_PATH,
        timestamp_col=config.TIMESTAMP_COL,
        normalization=config.NORMALIZATION,
        target_col=config.TARGET_COL,
    )

    # 2. Window (mode-dependent target alignment)
    X_windows, y = build_windows(
        X, y_raw,
        k=config.WINDOW_SIZE,
        mode=config.WINDOWING_MODE,
        target_idx=target_idx,
    )

    # 2b. Discretize target into classes (for homogeneity)
    y_classes, class_info = discretize_target(
        y, n_classes=config.N_Y_CLASSES, low_fraction=config.LOW_CLASS_FRACTION
    )

    # 3. Embedding
    Z, embed_meta = get_embeddings(X_windows, method=config.EMBEDDING_METHOD, y=y)
    print(f"\n[Embedding] Output shape: {Z.shape}")
    ae_history = embed_meta.get("history")   # None for non-AE methods

    # 3b. Optional embedding standardization
    if config.NORMALIZE_EMBEDDINGS:
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(Z).astype(np.float32)
        print("[Embedding] Standardized (z-scored) embedding dimensions before clustering.")

    # Per-window feature names (for FS importance plots)
    if config.WINDOWING_MODE == "latest_y":
        feat_no_target = [c for c in feature_names if c != config.TARGET_COL]
        window_feature_names = [f"{f}_t-{config.WINDOW_SIZE-1-i}"
                                for i in range(config.WINDOW_SIZE) for f in feat_no_target]
    else:
        window_feature_names = [f"{f}_t-{config.WINDOW_SIZE-1-i}"
                                for i in range(config.WINDOW_SIZE) for f in feature_names]

    # 4. Clustering
    cluster_params = {
        "kmeans_k":          config.KMEANS_K,
        "kmeans_init":       config.KMEANS_INIT,
        "kmeans_max_iter":   config.KMEANS_MAX_ITER,

        "dbscan_eps":        config.DBSCAN_EPS,
        "dbscan_min_samples":config.DBSCAN_MIN_SAMPLES,

        "agg_n_clusters":    config.AGG_N_CLUSTERS,
        "agg_linkage":       config.AGG_LINKAGE,
        "agg_connectivity":  config.AGG_CONNECTIVITY,
        "agg_connectivity_param": config.AGG_CONNECTIVITY_PARAM,

        "kmedoids_k":        config.KMEDOIDS_K,
        "kmedoids_metric":   config.KMEDOIDS_METRIC,
        "kmedoids_max_iter": config.KMEDOIDS_MAX_ITER,

        "gmm_n_components":  config.GMM_N_COMPONENTS,
        "gmm_covariance_type": config.GMM_COVARIANCE_TYPE,
    }
    clustering_results = run_clustering(
        Z,
        methods         = config.CLUSTERING_METHODS,
        distance_metric = config.DISTANCE_METRIC,
        params          = cluster_params,
        random_seed     = config.RANDOM_SEED,
    )

    # 5. Analysis
    print(f"\n{'='*70}")
    print("  Qualitative / Quantitative Analysis")
    print(f"{'='*70}")

    correlation_results = []
    external_results = []
    homogeneity_results = []
    summaries = []

    for alg_name, res in clustering_results.items():
        labels = res["labels"]
        print(f"\n--- {alg_name.upper()} ---")

        # Internal: embedding distance vs same-cluster indicator
        corr = pairwise_correlation_analysis(
            Z=Z, labels=labels, method=config.CORRELATION_METHOD,
            sample_pairs=config.ANALYSIS_SAMPLE_PAIRS, rng=rng, tag=alg_name,
        )
        corr["algorithm"] = alg_name
        correlation_results.append(corr)

        # External: embedding distance vs |Δy| (target-aware), intra & inter
        ext = external_correlation_analysis(
            Z=Z, y=y, labels=labels, method=config.CORRELATION_METHOD,
            max_pairs=config.EXT_MAX_PAIRS_PER_GROUP, rng=rng, tag=alg_name,
        )
        ext["algorithm"] = alg_name
        external_results.append(ext)

        # Homogeneity against discretized y-classes
        homo = homogeneity_analysis(labels, y_classes, tag=alg_name)
        homogeneity_results.append(homo)

        summary_df = cluster_summary(Z, labels, tag=alg_name)
        summaries.append(summary_df)

    compare_clusterings(clustering_results)

    # 5b. Supervised forecasting probe (frozen embeddings → y)
    supervised_results = None
    if config.RUN_SUPERVISED:
        sup_params = {
            "xgb_n_estimators": config.XGB_N_ESTIMATORS,
            "xgb_max_depth":    config.XGB_MAX_DEPTH,
            "xgb_learning_rate":config.XGB_LEARNING_RATE,
            "xgb_subsample":    config.XGB_SUBSAMPLE,

            "rf_n_estimators":  config.SUP_RF_N_ESTIMATORS,
            "rf_max_depth":     config.SUP_RF_MAX_DEPTH,
            
            "mlp_hidden_layers":config.MLP_HIDDEN_LAYERS,
            "mlp_alpha":        config.MLP_ALPHA,
            "mlp_learning_rate":config.MLP_LEARNING_RATE,
            "mlp_max_iter":     config.MLP_MAX_ITER,
        }
        supervised_results = run_supervised_probe(
            Z=Z, X_windows=X_windows, y=y,
            window_feature_names=window_feature_names,
            model_name=config.SUPERVISED_MODEL,
            model_params=sup_params,
            n_splits=config.CV_N_SPLITS,
            probe_on=config.SUPERVISED_ON,
            random_seed=config.RANDOM_SEED,
            run_shap=config.RUN_SHAP,
            shap_max_samples=config.SHAP_MAX_SAMPLES,
        )

    # 6. Save results
    config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}

    save_results(
        output_dir          = config.RESULTS_DIR,
        Z                   = Z,
        clustering_results  = clustering_results,
        correlation_results = correlation_results,
        summaries           = summaries,
        config_dict         = config_dict,
        y                   = y,
        y_classes           = y_classes,
        class_info          = class_info,
        external_results    = external_results,
        homogeneity_results = homogeneity_results,
        ae_history          = ae_history,
        embed_meta          = embed_meta,
        window_feature_names= window_feature_names,
        supervised_results  = supervised_results,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*70}")

    if config.PLOT:
        print("\n[Plotting] Generating plots from results")

        plot(config.RESULTS_DIR)

        print(f"[Plotting] Plots saved in {config.RESULTS_DIR}/plots")


if __name__ == "__main__":
    main()