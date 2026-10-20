"""
K-Means, DBSCAN, and Agglomerative clustering
with Euclidean or Mahalanobis distance. Methods are selected via a configurable list.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

def run_clustering(
    Z: np.ndarray,
    methods: list[str],
    distance_metric: str,
    params: dict,
    random_seed: int,
) -> dict:
    """
    Cluster the embedding matrix Z with one or more algorithms.

    Parameters
    ----------
    methods         : list of algorithm names, any of:
                      "kmeans", "dbscan", "agglomerative".
    distance_metric : "euclidean" or "mahalanobis" (applied via whitening).
    params          : dict of algorithm hyperparameters (see config). Keys used:
                      kmeans_k, kmeans_init, kmeans_max_iter,
                      dbscan_eps, dbscan_min_samples,
                      agg_n_clusters, agg_linkage.

    Returns
    -------
    results : dict  {method_name: {"labels": np.ndarray, "metrics": dict, "model": ...}}
    """
    print(f"\n[Clustering] Methods: {methods}  |  Distance: {distance_metric!r}  |  N={Z.shape[0]}")

    # Optionally transform to Mahalanobis space (whitening)
    Z_cluster = _prepare_metric_space(Z, distance_metric)

    dispatch = {
        "kmeans": lambda: _run_kmeans(
            Z_cluster, params["kmeans_k"], params["kmeans_init"],
            params["kmeans_max_iter"], random_seed, distance_metric),
        "dbscan": lambda: _run_dbscan(
            Z_cluster, params["dbscan_eps"], params["dbscan_min_samples"],
            random_seed, distance_metric),
        "agglomerative": lambda: _run_agglomerative(
            Z_cluster, params["agg_n_clusters"], params["agg_linkage"],
            distance_metric, params["agg_connectivity"],
            params["agg_connectivity_param"], random_seed),
        "kmedoids": lambda: _run_kmedoids(
            Z_cluster, params["kmedoids_k"], params["kmedoids_metric"],
            params["kmedoids_max_iter"], random_seed, distance_metric),
        "gmm": lambda: _run_gmm(
            Z_cluster, params["gmm_n_components"], params["gmm_covariance_type"],
            random_seed, distance_metric),
    }

    results = {}
    for m in methods:
        if m not in dispatch:
            raise ValueError(f"Unknown clustering method: {m!r}. "
                             f"Choose from {list(dispatch.keys())}.")
        results[m] = dispatch[m]()

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
    metrics = _compute_metrics(Z, labels, algorithm="kmeans", seed=seed)
    metrics["inertia"] = km.inertia_
    metrics["n_clusters"] = k
    print(f"[K-Means] Inertia={km.inertia_:.4f}  |  {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": km}


def _run_dbscan(Z, eps, min_samples, seed, metric_name):
    print(f"[DBSCAN] eps={eps}, min_samples={min_samples} …")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(Z)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"[DBSCAN] Clusters found: {n_clusters}  |  Noise points: {n_noise}")
    metrics = _compute_metrics(Z, labels, algorithm="dbscan", seed=seed)
    metrics["n_clusters"] = n_clusters
    metrics["n_noise"] = n_noise
    print(f"[DBSCAN] {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": db}


def _build_connectivity(Z, kind, param):
    """Build a sparse connectivity matrix to make agglomerative clustering
    memory-feasible on large N (avoids the O(n^2) full distance matrix).

    kind: "none"   -> no constraint (dense, O(n^2) memory)
          "knn"    -> k-nearest-neighbors graph (param = k, int)
          "radius" -> radius-neighbors graph (param = radius, float)
    """
    if kind == "none":
        return None
    from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
    if kind == "knn":
        return kneighbors_graph(Z, n_neighbors=int(param), include_self=False)
    if kind == "radius":
        return radius_neighbors_graph(Z, radius=float(param), include_self=False)
    raise ValueError(f"Unknown AGG_CONNECTIVITY: {kind!r}. Choose 'none', 'knn', or 'radius'.")


def _run_agglomerative(Z, n_clusters, linkage, metric_name,
                       connectivity="none", connectivity_param=15, seed=None):
    """
    Hierarchical agglomerative clustering (bottom-up merging).

    linkage: "ward" (minimizes within-cluster variance; requires Euclidean),
             "complete", "average", or "single".

    connectivity: a sparse neighbor graph that restricts which merges are allowed.
                  Using "knn" or "radius" avoids the dense O(n^2) distance matrix,
                  making large N feasible. "none" uses the full (memory-heavy) matrix.
    """
    print(f"[Agglomerative] n_clusters={n_clusters}, linkage={linkage!r}, "
          f"connectivity={connectivity!r}({connectivity_param}) …")
    if connectivity == "none" and Z.shape[0] > 50000:
        print(f"  [Warning] N={Z.shape[0]} with connectivity='none' needs O(n^2) memory. "
              f"Consider AGG_CONNECTIVITY='knn' or 'radius'.")

    conn = _build_connectivity(Z, connectivity, connectivity_param)
    if conn is not None:
        avg_neighbors = conn.nnz / Z.shape[0]
        if avg_neighbors < 1.0:
            print(f"  [Warning] connectivity graph is very sparse "
                  f"(~{avg_neighbors:.2f} neighbors/point). With '{connectivity}'="
                  f"{connectivity_param}, most points are isolated, which makes "
                  f"agglomerative slow and the result degenerate. Increase the parameter "
                  f"(for 'radius', try a value near the typical inter-point distance; "
                  f"for 'knn', try 10–30).")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, connectivity=conn)
    labels = agg.fit_predict(Z)
    metrics = _compute_metrics(Z, labels, algorithm="agglomerative", seed=seed)
    metrics["n_clusters"] = len(set(labels))
    print(f"[Agglomerative] {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": agg}


def _run_kmedoids(Z, k, metric, max_iter, seed, metric_name):
    """
    k-medoids (PAM / Voronoi-iteration). Cluster centers are actual data points
    (medoids), which makes it more robust to outliers than k-means and usable with
    any distance metric. Implemented directly (scikit-learn-extra is incompatible
    with numpy 2.x).
    """
    print(f"[K-Medoids] k={k}, metric={metric!r}, max_iter={max_iter} …")
    if Z.shape[0] > 30000:
        print(f"  [Warning] N={Z.shape[0]} is large for k-medoids; per-cluster pairwise "
              f"distances can be slow/memory-heavy.")
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    medoids = rng.choice(n, size=k, replace=False)
    D = pairwise_distances(Z, Z[medoids], metric=metric)
    labels = D.argmin(axis=1)

    for _ in range(max_iter):
        new_medoids = medoids.copy()
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            sub = pairwise_distances(Z[members], metric=metric)
            new_medoids[j] = members[sub.sum(axis=1).argmin()]
        D = pairwise_distances(Z, Z[new_medoids], metric=metric)
        new_labels = D.argmin(axis=1)
        if np.array_equal(new_medoids, medoids):
            labels = new_labels
            break
        medoids, labels = new_medoids, new_labels

    metrics = _compute_metrics(Z, labels, algorithm="kmedoids", seed=seed)
    metrics["n_clusters"] = k
    metrics["inertia"] = float(D[np.arange(n), labels].sum())  # total cost to medoids
    print(f"[K-Medoids] Cost={metrics['inertia']:.4f}  |  {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": {"medoids": medoids.tolist()}}


def _run_gmm(Z, n_components, covariance_type, seed, metric_name):
    """
    Gaussian Mixture Model. Models each cluster as an ellipsoidal Gaussian (vs.
    k-means' spherical assumption), which suits correlated autoencoder latents.
    Reports BIC/AIC for model selection across n_components.
    """
    print(f"[GMM] n_components={n_components}, covariance_type={covariance_type!r} …")
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type,
                          random_state=seed, n_init=3)
    labels = gmm.fit_predict(Z)
    metrics = _compute_metrics(Z, labels, algorithm="gmm", seed=seed)
    metrics["n_clusters"] = len(set(labels))
    metrics["bic"] = float(gmm.bic(Z))
    metrics["aic"] = float(gmm.aic(Z))
    print(f"[GMM] BIC={metrics['bic']:.1f}  AIC={metrics['aic']:.1f}  |  {_fmt_metrics(metrics)}")
    return {"labels": labels, "metrics": metrics, "model": gmm}


# Internal metrics

def _compute_metrics(Z: np.ndarray, labels: np.ndarray, algorithm: str, seed: int = None) -> dict:
    """Compute silhouette, Davies-Bouldin, Calinski-Harabasz where applicable."""
    metrics = {}
    unique = set(labels) - {-1}   # Exclude DBSCAN noise label
    if len(unique) >= 2:
        mask = labels != -1
        try:
            metrics["silhouette"]          = silhouette_score(Z[mask], labels[mask], random_state=seed)
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
