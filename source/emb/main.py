import os
import sys
import time
import numpy as np

# Make project root importable
sys.path.insert(0, os.path.dirname(__file__))

import config
from data_loader  import load_and_preprocess, build_windows
from embeddings   import get_embeddings
from clustering   import run_clustering
from analysis     import (
    pairwise_correlation_analysis,
    cluster_summary,
    compare_clusterings,
    save_results,
)


def main():
    t0 = time.time()
    rng = np.random.default_rng(config.RANDOM_SEED)

    print("=" * 70)
    print("  PV Energy Forecasting — Unsupervised Embedding Pipeline")
    print("=" * 70)

    # Load & normalise
    X, feature_names = load_and_preprocess(
        path=config.DATASET_PATH,
        timestamp_col=config.TIMESTAMP_COL,
        normalization=config.NORMALIZATION,
    )

    # Window
    X_windows = build_windows(X, k=config.WINDOW_SIZE)

    # Embedding
    Z, embed_meta = get_embeddings(X_windows, method=config.EMBEDDING_METHOD)
    print(f"\n[Embedding] Output shape: {Z.shape}")

    # Clustering
    clustering_results = run_clustering(
        Z,
        method           = config.CLUSTERING_METHOD,
        distance_metric  = config.DISTANCE_METRIC,
        kmeans_k         = config.KMEANS_K,
        kmeans_init      = config.KMEANS_INIT,
        kmeans_max_iter  = config.KMEANS_MAX_ITER,
        dbscan_eps       = config.DBSCAN_EPS,
        dbscan_min_samples = config.DBSCAN_MIN_SAMPLES,
        random_seed      = config.RANDOM_SEED,
    )

    # Analysis
    print(f"\n{'='*70}")
    print("  Qualitative / Quantitative Analysis")
    print(f"{'='*70}")

    correlation_results = []
    summaries = []

    for alg_name, res in clustering_results.items():
        labels = res["labels"]
        print(f"\n--- Correlation analysis: {alg_name.upper()} ---")
        corr = pairwise_correlation_analysis(
            Z=Z,
            labels=labels,
            method=config.CORRELATION_METHOD,
            sample_pairs=config.ANALYSIS_SAMPLE_PAIRS,
            rng=rng,
            tag=alg_name,
        )
        corr["algorithm"] = alg_name
        correlation_results.append(corr)

        summary_df = cluster_summary(Z, labels, tag=alg_name)
        summaries.append(summary_df)

    compare_clusterings(clustering_results)

    # Save results
    # Build a serialisable config snapshot
    config_dict = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}

    save_results(
        output_dir        = config.RESULTS_DIR,
        Z                 = Z,
        clustering_results= clustering_results,
        correlation_results= correlation_results,
        summaries         = summaries,
        config_dict       = config_dict,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
