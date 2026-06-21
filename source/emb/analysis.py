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
from itertools import combinations
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, homogeneity_score


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


# External (target-aware) intra/inter correlation

def _corr(d: np.ndarray, dy: np.ndarray, method: str) -> tuple[float, float]:
    """Correlation between embedding distances d and target gaps dy.
       Returns (r, p). If computation not possible returns (nan, nan).
    """
    if len(d) < 2 or np.std(d) == 0 or np.std(dy) == 0:
        return float("nan"), float("nan")
    if method == "pearson":
        r, p = stats.pearsonr(d, dy)
    else:
        r, p = stats.spearmanr(d, dy)
    return float(r), float(p)


def _sample_pairs_within(idx: np.ndarray, max_pairs: int | None, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays (a, b) of unordered within-group pairs, optionally subsampled."""
    n = len(idx)
    total = n * (n - 1) // 2
    if max_pairs is None or total <= max_pairs:
        pairs = np.array(list(combinations(idx, 2)))
        return pairs[:, 0], pairs[:, 1]
    # Subsample: draw random distinct pairs
    a = rng.choice(idx, size=max_pairs, replace=True)
    b = rng.choice(idx, size=max_pairs, replace=True)
    keep = a != b
    return a[keep], b[keep]


def _sample_pairs_between(idx_i: np.ndarray, idx_j: np.ndarray, max_pairs: int | None, rng):
    """Return arrays (a, b) of cross-group pairs (a from i, b from j), optionally subsampled."""
    total = len(idx_i) * len(idx_j)
    if max_pairs is None or total <= max_pairs:
        a = np.repeat(idx_i, len(idx_j))
        b = np.tile(idx_j, len(idx_i))
        return a, b
    a = rng.choice(idx_i, size=max_pairs, replace=True)
    b = rng.choice(idx_j, size=max_pairs, replace=True)
    return a, b


def external_correlation_analysis(
    Z: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    method: str,
    max_pairs: int | None,
    rng: np.random.Generator,
    tag: str = "",
) -> dict:
    """
    External intra/inter cluster correlation between embedding distance and |Δy|.

    IntraCorr(c_i) = corr( {||z_a - z_b|| : a,b in c_i},  {|y_a - y_b| : a,b in c_i} )
    IntraCorr(c)   = mean over clusters of IntraCorr(c_i)

    InterCorr(c_i,c_j) = corr over all cross pairs (a in c_i, b in c_j)
    InterCorr(c)       = mean over all cluster pairs

    A positive correlation means: pairs that are farther apart in embedding space
    also differ more in solar generation — i.e. the embedding geometry tracks the
    target. Reported for both Pearson and Spearman.

    Returns a dict with per-cluster and aggregate values.
    """
    clusters = sorted(set(labels) - {-1})   # exclude DBSCAN noise
    if len(clusters) < 1:
        print(f"  [{tag}] No valid clusters for external correlation. Skipping.")
        return {}

    idx_by_cluster = {c: np.where(labels == c)[0] for c in clusters}

    # Intra
    intra_pearson, intra_pearson_p = {}, {}
    intra_spearman, intra_spearman_p = {}, {}
    for c in clusters:
        idx = idx_by_cluster[c]
        if len(idx) < 2:
            continue
        a, b = _sample_pairs_within(idx, max_pairs, rng)
        if len(a) < 2:
            continue
        d  = np.linalg.norm(Z[a] - Z[b], axis=1)
        dy = np.abs(y[a] - y[b])
        r, p = _corr(d, dy, "pearson")
        intra_pearson[c] = r
        intra_pearson_p[c] = p
        r2, p2 = _corr(d, dy, "spearman")
        intra_spearman[c] = r2
        intra_spearman_p[c] = p2

    # Inter
    inter_pearson, inter_pearson_p = {}, {}
    inter_spearman, inter_spearman_p = {}, {}
    for ci, cj in combinations(clusters, 2):
        a, b = _sample_pairs_between(idx_by_cluster[ci], idx_by_cluster[cj], max_pairs, rng)
        if len(a) < 2:
            continue
        d  = np.linalg.norm(Z[a] - Z[b], axis=1)
        dy = np.abs(y[a] - y[b])
        r, p = _corr(d, dy, "pearson")
        inter_pearson[(ci, cj)] = r
        inter_pearson_p[(ci, cj)] = p
        r2, p2 = _corr(d, dy, "spearman")
        inter_spearman[(ci, cj)] = r2
        inter_spearman_p[(ci, cj)] = p2

    result = {
        "tag": tag,
        "n_clusters_used": len(clusters),
        "intra_corr_pearson_mean":  float(np.nanmean(list(intra_pearson.values())))  if intra_pearson  else float("nan"),
        "intra_corr_pearson_mean_p": float(np.nanmean(list(intra_pearson_p.values()))) if intra_pearson_p else float("nan"),
        "intra_corr_spearman_mean": float(np.nanmean(list(intra_spearman.values()))) if intra_spearman else float("nan"),
        "intra_corr_spearman_mean_p": float(np.nanmean(list(intra_spearman_p.values()))) if intra_spearman_p else float("nan"),
        "inter_corr_pearson_mean":  float(np.nanmean(list(inter_pearson.values())))  if inter_pearson  else float("nan"),
        "inter_corr_pearson_mean_p": float(np.nanmean(list(inter_pearson_p.values()))) if inter_pearson_p else float("nan"),
        "inter_corr_spearman_mean": float(np.nanmean(list(inter_spearman.values()))) if inter_spearman else float("nan"),
        "inter_corr_spearman_mean_p": float(np.nanmean(list(inter_spearman_p.values()))) if inter_spearman_p else float("nan"),
        "intra_per_cluster_pearson":  {int(k): v for k, v in intra_pearson.items()},
        "intra_per_cluster_pearson_p": {int(k): v for k, v in intra_pearson_p.items()},
        "intra_per_cluster_spearman": {int(k): v for k, v in intra_spearman.items()},
        "intra_per_cluster_spearman_p": {int(k): v for k, v in intra_spearman_p.items()},
        # "inter_per_pair_pearson":  {str(k): v for k, v in inter_pearson.items()},
        # "inter_per_pair_pearson_p": {str(k): v for k, v in inter_pearson_p.items()},
        # "inter_per_pair_spearman": {str(k): v for k, v in inter_spearman.items()},
        # "inter_per_pair_spearman_p": {str(k): v for k, v in inter_spearman_p.items()},
    }

    print(f"  [{tag}] External IntraCorr  Pearson={result['intra_corr_pearson_mean']:.4f}  "
          f"Spearman={result['intra_corr_spearman_mean']:.4f}")
    print(f"  [{tag}] External InterCorr  Pearson={result['inter_corr_pearson_mean']:.4f}  "
          f"Spearman={result['inter_corr_spearman_mean']:.4f}")
    return result


# Homogeneity (against discretized y classes)

def homogeneity_analysis(labels: np.ndarray, y_classes: np.ndarray, tag: str = "") -> dict:
    """
    Homogeneity score: each cluster should contain only members of a single y-class.
    Range [0, 1], higher is better. Noise points (-1) are excluded.
    """
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask])) < 1:
        print(f"  [{tag}] Not enough non-noise samples for homogeneity. Skipping.")
        return {"tag": tag, "homogeneity": float("nan")}

    h = homogeneity_score(y_classes[mask], labels[mask])
    print(f"  [{tag}] Homogeneity (vs y-classes) = {h:.4f}")
    return {"tag": tag, "homogeneity": float(h)}


# Persist results

def save_results(
    output_dir: str,
    Z: np.ndarray,
    clustering_results: dict,
    correlation_results: list[dict],
    summaries: list[pd.DataFrame],
    config_dict: dict,
    y: np.ndarray = None,
    y_classes: np.ndarray = None,
    class_info: dict = None,
    external_results: list[dict] = None,
    homogeneity_results: list[dict] = None,
    ae_history: dict = None,
    embed_meta: dict = None,
    window_feature_names: list[str] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), Z)

    # External target y and its discretized classes
    if y is not None:
        np.save(os.path.join(output_dir, "y_targets.npy"), y)
    if y_classes is not None:
        np.save(os.path.join(output_dir, "y_classes.npy"), y_classes)
    if class_info is not None:
        with open(os.path.join(output_dir, "y_class_info.json"), "w") as f:
            json.dump(class_info, f, indent=2)

    # Labels for each clustering
    for name, res in clustering_results.items():
        np.save(os.path.join(output_dir, f"labels_{name}.npy"), res["labels"])

    # Cluster metrics (merge in homogeneity per algorithm)
    homo_by_alg = {r["tag"]: r.get("homogeneity") for r in (homogeneity_results or [])}
    metrics_rows = []
    for name, res in clustering_results.items():
        row = {"algorithm": name}
        row.update(res["metrics"])
        if name in homo_by_alg:
            row["homogeneity"] = homo_by_alg[name]
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(output_dir, "cluster_metrics.csv"), index=False)

    # Correlation analysis (geometry-vs-cluster)
    with open(os.path.join(output_dir, "correlation_analysis.json"), "w") as f:
        json.dump(_jsonify(correlation_results), f, indent=2)

    # External (target-aware) correlation analysis
    if external_results is not None:
        with open(os.path.join(output_dir, "external_correlation_analysis.json"), "w") as f:
            json.dump(_jsonify(external_results), f, indent=2)

    # Autoencoder training history
    if ae_history is not None:
        with open(os.path.join(output_dir, "ae_training_history.json"), "w") as f:
            json.dump(_jsonify([ae_history])[0], f, indent=2)

    # Cluster summaries
    for i, df in enumerate(summaries):
        df.to_csv(os.path.join(output_dir, f"cluster_summary_{i}.csv"), index=False)

    # Window feature names (for FS importance plots)
    if window_feature_names is not None:
        with open(os.path.join(output_dir, "window_feature_names.json"), "w") as f:
            json.dump(window_feature_names, f)

    # Embedding meta (selected indices / importances), skipping the torch model
    if embed_meta is not None:
        meta_clean = {}
        for k, v in embed_meta.items():
            if k in ("model", "history"):
                continue
            if isinstance(v, np.ndarray):
                meta_clean[k] = v.tolist()
            elif not callable(v):
                meta_clean[k] = v
        with open(os.path.join(output_dir, "embed_meta.json"), "w") as f:
            json.dump(meta_clean, f, indent=2)

    # Config snapshot
    serialisable = {k: (v if not callable(v) else str(v)) for k, v in config_dict.items()}
    with open(os.path.join(output_dir, "config_snapshot.json"), "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\n[Results] Saved to '{output_dir}/'")
    print("  embeddings.npy, y_targets.npy, y_classes.npy, y_class_info.json,")
    print("  labels_*.npy, cluster_metrics.csv, correlation_analysis.json,")
    print("  external_correlation_analysis.json, ae_training_history.json,")
    print("  cluster_summary_*.csv, embed_meta.json, config_snapshot.json")


def _jsonify(obj):
    """Recursively convert numpy types and tuple-keyed dicts to JSON-serialisable forms."""
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = str(k) if isinstance(k, tuple) else k
            out[key] = _jsonify(v)
        return out
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
