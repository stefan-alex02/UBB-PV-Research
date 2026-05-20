"""
K-Means and DBSCAN clustering with Euclidean or Mahalanobis distance.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def run_clustering(
    Z: np.ndarray,
    method: str,
    distance_metric: str,
    kmeans_k: int,
    kmeans_init: str,
    kmeans_max_iter: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    random_seed: int,
) -> dict:
    """
    Cluster the embedding matrix Z.

    Returns
    -------
    results : dict  {method_name: {"labels": np.ndarray, "metrics": dict}}
    """
    print(f"\n[Clustering] Method: {method!r}  |  Distance: {distance_metric!r}  |  N={Z.shape[0]}")

    # Optionally transform to Mahalanobis space (whitening)
    Z_cluster = _prepare_metric_space(Z, distance_metric)

    results = {}
    if method in ("kmeans", "both"):
        results["kmeans"] = _run_kmeans(Z_cluster, kmeans_k, kmeans_init, kmeans_max_iter, random_seed, distance_metric)
    if method in ("dbscan", "both"):
        results["dbscan"] = _run_dbscan(Z_cluster, dbscan_eps, dbscan_min_samples, random_seed, distance_metric)

    return results


def _prepare_metric_space(Z: np.ndarray, metric: str) -> np.ndarray:
    """
    For Mahalanobis: whiten the data (multiply by Σ^{-1/2}) so that
    Euclidean distance in the whitened space equals Mahalanobis in original space.
    This makes standard sklearn clusterers compatible.
    """
    if metric == "euclidean":
        return Z
    elif metric == "mahalanobis":
        print("[Clustering] Whitening data for Mahalanobis-equivalent distance …")
        cov = np.cov(Z, rowvar=False)
        # Regularise to avoid singular covariance
        cov += np.eye(cov.shape[0]) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-10)
        W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return Z @ W.T
    else:
        raise ValueError(f"Unknown distance metric: {metric!r}. Choose 'euclidean' or 'mahalanobis'.")


def _run_kmeans(Z, k, init, max_iter, seed, metric_name):
    print(f"[K-Means] k={k}, init={init!r}, max_iter={max_iter} …")
    km = KMeans(n_clusters=k, init=init, max_iter=max_iter, random_state=seed, n_init=10)
    labels = km.fit_predict(Z)
    metrics = _compute_metrics(Z, labels, algorithm="kmeans")
    metrics["inertia"] = km.inertia_
    print(f"[K-Means] Inertia={km.inertia_:.4f}  |  {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": km}


def _run_dbscan(Z, eps, min_samples, seed, metric_name):
    print(f"[DBSCAN] eps={eps}, min_samples={min_samples} …")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(Z)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[DBSCAN] Clusters found: {n_clusters}  |  Noise points: {n_noise}")
    metrics = _compute_metrics(Z, labels, algorithm="dbscan")
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = n_noise
    print(f"[DBSCAN] {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": db}


def _compute_metrics(Z: np.ndarray, labels: np.ndarray, algorithm: str) -> dict:
    """Compute silhouette, Davies-Bouldin, Calinski-Harabasz where applicable."""
    metrics = {}
    unique = set(labels) - {-1}   # Exclude DBSCAN noise label
    if len(unique) >= 2:
        mask = labels != -1
        try:
            metrics["silhouette"]          = silhouette_score(Z[mask], labels[mask], sample_size=min(5000, mask.sum()))
            metrics["davies_bouldin"]      = davies_bouldin_score(Z[mask], labels[mask])
            metrics["calinski_harabasz"]   = calinski_harabasz_score(Z[mask], labels[mask])
        except Exception as e:
            print(f"  [Warning] Could not compute some metrics: {e}")
    else:
        print(f"  [Warning] Only {len(unique)} cluster(s) found — skipping cluster quality metrics.")
    return metrics


def _fmt_metrics(m: dict) -> str:
    parts = []
    if "silhouette"        in m: parts.append(f"Silhouette={m['silhouette']:.4f}")
    if "davies_bouldin"    in m: parts.append(f"DB={m['davies_bouldin']:.4f}")
    if "calinski_harabasz" in m: parts.append(f"CH={m['calinski_harabasz']:.2f}")
    return "  |  ".join(parts) if parts else "N/A"
