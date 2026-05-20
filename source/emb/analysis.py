"""
Qualitative & quantitative analysis:

1. Pairwise embedding distance vs. cluster-assignment distance (Pearson/Spearman).
2. Summary statistics per cluster.
3. Save all results to disk.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score


def pairwise_correlation_analysis(
    Z: np.ndarray,
    labels: np.ndarray,
    method: str,
    sample_pairs: int | None,
    rng: np.random.Generator,
    tag: str = "",
) -> dict:
    """
    Test whether samples that are close in embedding space also share the same cluster.

    Strategy
    --------
    1. Draw `sample_pairs` random pairs of samples.
    2. Compute their Euclidean distance in the embedding space.
    3. Compute a binary "same cluster" indicator (1 = same, 0 = different; noise=-1 excluded).
    4. Compute Pearson / Spearman correlation between (distances, same-cluster-indicator).
       - Negative correlation expected: closer → more likely same cluster.

    Additionally, compute the ratio of intra-cluster vs. inter-cluster mean distance
    as a compactness measure.
    """
    N = len(Z)
    # Exclude noise labels (-1) from analysis
    valid_mask = labels != -1
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) < 10:
        print(f"  [{tag}] Not enough non-noise samples for analysis ({len(valid_idx)}). Skipping.")
        return {}

    # Sample pairs
    n_pairs = min(sample_pairs or len(valid_idx) ** 2, len(valid_idx) * (len(valid_idx) - 1) // 2)
    n_pairs = min(n_pairs, 50_000)  # hard cap for speed

    idx_a = rng.choice(valid_idx, size=n_pairs, replace=True)
    idx_b = rng.choice(valid_idx, size=n_pairs, replace=True)
    # Avoid self-pairs
    different = idx_a != idx_b
    idx_a, idx_b = idx_a[different], idx_b[different]

    # Embedding distances
    diffs = Z[idx_a] - Z[idx_b]
    emb_distances = np.linalg.norm(diffs, axis=1)

    # Same-cluster indicator
    same_cluster = (labels[idx_a] == labels[idx_b]).astype(float)

    results = {"tag": tag, "n_pairs": len(emb_distances)}

    if method in ("pearson", "both"):
        r, p = stats.pearsonr(emb_distances, same_cluster)
        results["pearson_r"]  = r
        results["pearson_p"]  = p
        print(f"  [{tag}] Pearson r={r:.4f}  (p={p:.3e})")

    if method in ("spearman", "both"):
        rho, p = stats.spearmanr(emb_distances, same_cluster)
        results["spearman_rho"] = rho
        results["spearman_p"]   = p
        print(f"  [{tag}] Spearman ρ={rho:.4f}  (p={p:.3e})")

    # Intra vs inter cluster distance ratio
    intra_dists = emb_distances[same_cluster == 1]
    inter_dists = emb_distances[same_cluster == 0]
    if len(intra_dists) > 0 and len(inter_dists) > 0:
        ratio = intra_dists.mean() / inter_dists.mean()
        results["intra_mean_dist"]  = float(intra_dists.mean())
        results["inter_mean_dist"]  = float(inter_dists.mean())
        results["intra_inter_ratio"]= float(ratio)
        print(f"  [{tag}] Intra-cluster dist mean={intra_dists.mean():.4f}  "
              f"Inter-cluster dist mean={inter_dists.mean():.4f}  "
              f"Ratio={ratio:.4f}  (lower is better)")

    return results


def cluster_summary(Z: np.ndarray, labels: np.ndarray, tag: str = "") -> pd.DataFrame:
    """Return a DataFrame with per-cluster size and centroid norm."""
    rows = []
    for lbl in sorted(set(labels)):
        mask = labels == lbl
        cluster_Z = Z[mask]
        centroid = cluster_Z.mean(axis=0)
        rows.append({
            "cluster": int(lbl),
            "size": int(mask.sum()),
            "centroid_norm": float(np.linalg.norm(centroid)),
            "intra_std_mean": float(cluster_Z.std(axis=0).mean()),
        })
    df = pd.DataFrame(rows)
    print(f"\n  [{tag}] Cluster summary:\n{df.to_string(index=False)}\n")
    return df


def compare_clusterings(all_results: dict) -> None:
    """Print Adjusted Rand Index between every pair of clustering results."""
    keys = list(all_results.keys())
    if len(keys) < 2:
        return
    print("\n[Analysis] Adjusted Rand Index between clusterings:")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            la, lb = all_results[keys[i]]["labels"], all_results[keys[j]]["labels"]
            # ARI ignores noise labels: replace -1 with a unique label per point
            la_clean = _clean_labels(la)
            lb_clean = _clean_labels(lb)
            ari = adjusted_rand_score(la_clean, lb_clean)
            print(f"  ARI({keys[i]}, {keys[j]}) = {ari:.4f}")


def _clean_labels(labels: np.ndarray) -> np.ndarray:
    out = labels.copy()
    max_lbl = out.max()
    for i, l in enumerate(out):
        if l == -1:
            max_lbl += 1
            out[i] = max_lbl
    return out


def save_results(
    output_dir: str,
    Z: np.ndarray,
    clustering_results: dict,
    correlation_results: list[dict],
    summaries: list[pd.DataFrame],
    config_dict: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), Z)

    # Labels for each clustering
    for name, res in clustering_results.items():
        np.save(os.path.join(output_dir, f"labels_{name}.npy"), res["labels"])

    # Cluster metrics
    metrics_rows = []
    for name, res in clustering_results.items():
        row = {"algorithm": name}
        row.update(res["metrics"])
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(output_dir, "cluster_metrics.csv"), index=False)

    # Correlation analysis
    corr_path = os.path.join(output_dir, "correlation_analysis.json")
    # Convert numpy floats to python floats for JSON serialisation
    clean = []
    for r in correlation_results:
        clean.append({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()})
    with open(corr_path, "w") as f:
        json.dump(clean, f, indent=2)

    # Cluster summaries
    for i, df in enumerate(summaries):
        df.to_csv(os.path.join(output_dir, f"cluster_summary_{i}.csv"), index=False)

    # Config snapshot
    cfg_path = os.path.join(output_dir, "config_snapshot.json")
    serialisable = {k: (v if not callable(v) else str(v)) for k, v in config_dict.items()}
    with open(cfg_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\n[Results] Saved '{output_dir}/'")
