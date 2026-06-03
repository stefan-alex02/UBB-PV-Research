"""
plot_results.py — Visualise the outputs of the PV embedding pipeline.

Usage
-----
    python plot_results.py                        # uses default RESULTS_DIR = "results"
    python plot_results.py path/to/results_dir    # custom path

Produces one PNG per figure inside <results_dir>/plots/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

matplotlib.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

COLORMAP = "tab10"

results_dir = "results/ae/run2_higher_dbscan_eps"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> dict:
    data = {"dir": results_dir}

    path = lambda f: os.path.join(results_dir, f)

    data["Z"] = np.load(path("embeddings.npy"))

    data["labels"] = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("labels_") and fname.endswith(".npy"):
            key = fname[len("labels_"):-len(".npy")]
            data["labels"][key] = np.load(path(fname))

    data["cluster_metrics"] = pd.read_csv(path("cluster_metrics.csv"))

    with open(path("correlation_analysis.json")) as f:
        data["correlation"] = json.load(f)

    data["summaries"] = {}
    for fname in os.listdir(results_dir):
        if fname.startswith("cluster_summary_") and fname.endswith(".csv"):
            idx = fname[len("cluster_summary_"):-len(".csv")]
            data["summaries"][idx] = pd.read_csv(path(fname))

    with open(path("config_snapshot.json")) as f:
        data["config"] = json.load(f)

    meta_path = path("embed_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            data["embed_meta"] = json.load(f)
    else:
        data["embed_meta"] = {}

    names_path = path("window_feature_names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            data["window_feature_names"] = json.load(f)
    else:
        data["window_feature_names"] = None

    return data


def out(results_dir: str, name: str) -> str:
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, name)


def label_colors(labels, cmap=COLORMAP):
    unique = sorted(set(labels))
    cm = plt.get_cmap(cmap, max(len(unique), 2))
    return {lbl: ("lightgrey" if lbl == -1 else cm(i)) for i, lbl in enumerate(unique)}


# ─── 1. PCA + t-SNE 2-D projections coloured by cluster ──────────────────────

def plot_projections(d: dict):
    Z = d["Z"]
    labels_dict = d["labels"]
    cfg = d["config"]
    emb_method = cfg.get("EMBEDDING_METHOD", "")

    # Compute PCA (fast) and t-SNE (slower but better topology)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(Z)
    print("  Computing t-SNE …")
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500).fit_transform(Z)

    for alg, labels in labels_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"2-D Projections — embedding: {emb_method}, clustering: {alg.upper()}",
            fontsize=12, fontweight="bold"
        )

        for ax, proj, title in zip(axes, [pca2, tsne2], ["PCA", "t-SNE"]):
            cmap = label_colors(labels)
            colors = [cmap[l] for l in labels]
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=8, alpha=0.6, linewidths=0)
            ax.set_title(title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            unique_labels = sorted(set(labels))
            if len(unique_labels) <= 20:
                legend_els = [
                    Patch(color=cmap[l], label="Noise" if l == -1 else f"C{l}")
                    for l in unique_labels
                ]
                ax.legend(handles=legend_els, fontsize=7, markerscale=1.5,
                          ncol=max(1, len(legend_els) // 10), loc="best")
            else:
                # Too many clusters to label individually — just note the count
                n_noise = (labels == -1).sum()
                ax.set_xlabel(f"Component 1  ({len(unique_labels) - 1} clusters, "
                              f"{n_noise} noise pts)")

        plt.tight_layout()
        fpath = out(d["dir"], f"projections_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# ─── 2. Cluster size distribution bar chart ───────────────────────────────────

def plot_cluster_sizes(d: dict):
    summaries = d["summaries"]
    alg_names = list(d["labels"].keys())   # preserves insertion order
    n = len(summaries)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (idx, df) in zip(axes, summaries.items()):
        alg = alg_names[int(idx)] if int(idx) < len(alg_names) else idx
        cmap = plt.get_cmap(COLORMAP, max(len(df), 2))
        colors = ["lightgrey" if r["cluster"] == -1 else cmap(i % 10)
                  for i, r in df.iterrows()]

        n_clusters = len(df)
        # For many clusters (DBSCAN), use a horizontal bar chart so labels fit
        if n_clusters > 15:
            top_df = df.nlargest(20, "size")   # show only top-20 for readability
            top_colors = [("lightgrey" if r["cluster"] == -1
                           else cmap(i % 10))
                          for i, r in top_df.iterrows()]
            ax.barh([str(c) for c in top_df["cluster"]], top_df["size"],
                    color=top_colors, edgecolor="white")
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Cluster")
            ax.set_title(f"Cluster sizes (top 20) — {alg.upper()}\n"
                         f"Total clusters: {n_clusters}  |  "
                         f"Noise: {int(df.loc[df['cluster']==-1,'size'].sum()) if -1 in df['cluster'].values else 0}",
                         fontweight="bold")
        else:
            bars = ax.bar([str(c) for c in df["cluster"]], df["size"],
                          color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%d", fontsize=8)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"Cluster sizes — {alg.upper()}", fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "cluster_sizes.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 3. Silhouette plot per sample ────────────────────────────────────────────

def plot_silhouette(d: dict):
    Z = d["Z"]
    for alg, labels in d["labels"].items():
        valid = labels != -1
        if valid.sum() < 2 or len(set(labels[valid])) < 2:
            continue

        Z_v, lbl_v = Z[valid], labels[valid]
        sample_sil = silhouette_samples(Z_v, lbl_v)

        unique_labels = sorted(set(lbl_v))
        cmap = plt.get_cmap(COLORMAP, max(len(unique_labels), 2))

        fig, ax = plt.subplots(figsize=(8, max(4, len(unique_labels) * 0.8)))
        y_lower = 10
        yticks, yticklabels = [], []

        for i, lbl in enumerate(unique_labels):
            vals = np.sort(sample_sil[lbl_v == lbl])
            size = len(vals)
            y_upper = y_lower + size
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                             facecolor=cmap(i), alpha=0.75)
            yticks.append(y_lower + size / 2)
            yticklabels.append(f"C{lbl} (n={size})")
            y_lower = y_upper + 5

        mean_sil = sample_sil.mean()
        ax.axvline(mean_sil, color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean = {mean_sil:.3f}")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, fontsize=8)
        ax.set_xlabel("Silhouette coefficient")
        ax.set_title(f"Silhouette analysis — {alg.upper()}", fontweight="bold")
        ax.legend()

        plt.tight_layout()
        fpath = out(d["dir"], f"silhouette_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# ─── 4. Cluster quality metrics bar chart ─────────────────────────────────────

def plot_cluster_metrics(d: dict):
    df = d["cluster_metrics"]
    metric_cols = [c for c in ["silhouette", "davies_bouldin", "calinski_harabasz"]
                   if c in df.columns]
    if not metric_cols:
        return

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    notes = {
        "silhouette":          "↑ higher is better  (−1 to 1)",
        "davies_bouldin":      "↓ lower is better",
        "calinski_harabasz":   "↑ higher is better",
    }

    for ax, col in zip(axes, metric_cols):
        vals = df[col].astype(float)
        bars = ax.bar(df["algorithm"], vals,
                      color=["#4C72B0", "#DD8452"][:len(df)], edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=9)
        ax.set_title(f"{col.replace('_', ' ').title()}\n{notes.get(col,'')}", fontweight="bold")
        ax.set_ylabel(col)

    plt.tight_layout()
    fpath = out(d["dir"], "cluster_metrics.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 5. Intra vs inter cluster distance distributions ─────────────────────────

def plot_distance_distributions(d: dict, n_pairs: int = 4000, seed: int = 42):
    Z = d["Z"]
    rng = np.random.default_rng(seed)

    for alg, labels in d["labels"].items():
        valid_idx = np.where(labels != -1)[0]
        if len(valid_idx) < 20:
            continue

        idx_a = rng.choice(valid_idx, size=n_pairs, replace=True)
        idx_b = rng.choice(valid_idx, size=n_pairs, replace=True)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]

        dists = np.linalg.norm(Z[idx_a] - Z[idx_b], axis=1)
        same  = labels[idx_a] == labels[idx_b]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dists[same],  bins=60, alpha=0.65, label="Intra-cluster",  color="#4C72B0", density=True)
        ax.hist(dists[~same], bins=60, alpha=0.65, label="Inter-cluster",  color="#DD8452", density=True)
        ax.set_xlabel("Euclidean distance in embedding space")
        ax.set_ylabel("Density")
        ax.set_title(f"Distance distribution — {alg.upper()}", fontweight="bold")
        ax.legend()

        plt.tight_layout()
        fpath = out(d["dir"], f"distance_dist_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# ─── 6. Correlation scatter: embedding distance vs same-cluster indicator ─────

def plot_correlation_summary(d: dict):
    corr_list = d["correlation"]
    if not corr_list:
        return

    rows = []
    for r in corr_list:
        row = {"algorithm": r.get("algorithm", r.get("tag", "?"))}
        if "pearson_r"    in r: row["Pearson r"]   = r["pearson_r"]
        if "spearman_rho" in r: row["Spearman ρ"]  = r["spearman_rho"]
        if "intra_inter_ratio" in r: row["Intra/Inter ratio"] = r["intra_inter_ratio"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("algorithm")
    metric_cols = [c for c in ["Pearson r", "Spearman ρ", "Intra/Inter ratio"] if c in df.columns]

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 4))
    if len(metric_cols) == 1:
        axes = [axes]

    notes = {
        "Pearson r":         "↓ more negative → distances reflect clusters better",
        "Spearman ρ":        "↓ more negative → distances reflect clusters better",
        "Intra/Inter ratio": "↓ lower → more compact / separated clusters",
    }

    for ax, col in zip(axes, metric_cols):
        vals = df[col].astype(float)
        bars = ax.bar(df.index, vals,
                      color=["#4C72B0", "#DD8452"][:len(df)], edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{col}\n{notes.get(col,'')}", fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "correlation_summary.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 7. Feature importance (for FS-based embedding methods) ───────────────────

def plot_feature_importance(d: dict):
    meta = d.get("embed_meta", {})
    method = meta.get("method", d["config"].get("EMBEDDING_METHOD", ""))
    feat_names = d.get("window_feature_names")

    importance_key = None
    if method == "random_forest_fs" and "importances" in meta:
        importance_key = "importances"
        title = "Random Forest Feature Importances"
    elif method == "relieff" and "feature_importances" in meta:
        importance_key = "feature_importances"
        title = "ReliefF Feature Weights"
    elif method == "univariate_fs" and "scores" in meta:
        importance_key = "scores"
        title = "Univariate FS — F-scores"
    else:
        return  # not a FS method or data not saved

    scores = np.array(meta[importance_key])
    top_n = min(40, len(scores))
    top_idx = np.argsort(scores)[::-1][:top_n]
    top_scores = scores[top_idx]
    top_labels = [feat_names[i] if feat_names else f"feat_{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(11, max(4, top_n * 0.28)))
    bars = ax.barh(range(top_n), top_scores[::-1], color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=8)
    ax.set_xlabel("Score / Importance")
    ax.set_title(f"{title} — top {top_n}", fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "feature_importance.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 8. Embedding dimension statistics (mean ± std per dimension) ─────────────

def plot_embedding_stats(d: dict):
    Z = d["Z"]
    means = Z.mean(axis=0)
    stds  = Z.std(axis=0)
    idx = np.arange(len(means))

    fig, axes = plt.subplots(2, 1, figsize=(max(8, len(means) * 0.3), 7), sharex=True)

    axes[0].bar(idx, means, color="#4C72B0", edgecolor="none", alpha=0.85)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Mean activation")
    axes[0].set_title("Embedding dimension statistics", fontweight="bold")

    axes[1].bar(idx, stds, color="#DD8452", edgecolor="none", alpha=0.85)
    axes[1].set_ylabel("Std deviation")
    axes[1].set_xlabel("Embedding dimension")

    plt.tight_layout()
    fpath = out(d["dir"], "embedding_stats.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 9. Embedding correlation heatmap ─────────────────────────────────────────

def plot_embedding_correlation(d: dict):
    Z = d["Z"]
    if Z.shape[1] > 64:
        print(f"  [Skip] Embedding dim {Z.shape[1]} > 64 — skipping correlation heatmap.")
        return

    corr_mat = np.corrcoef(Z.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr_mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Embedding dimension pairwise correlation", fontweight="bold")
    ax.set_xlabel("Dimension index")
    ax.set_ylabel("Dimension index")

    plt.tight_layout()
    fpath = out(d["dir"], "embedding_correlation.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# ─── 10. Summary statistics table ─────────────────────────────────────────────

def print_summary_table(d: dict):
    cfg = d["config"]
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Embedding method  : {cfg.get('EMBEDDING_METHOD')}")
    print(f"  Window size (k)   : {cfg.get('WINDOW_SIZE')}")
    print(f"  Normalization     : {cfg.get('NORMALIZATION')}")
    print(f"  Distance metric   : {cfg.get('DISTANCE_METRIC')}")
    print(f"  Embedding shape   : {d['Z'].shape}")
    print()

    print("  Clustering metrics:")
    print(d["cluster_metrics"].to_string(index=False))
    print()

    print("  Correlation analysis:")
    for r in d["correlation"]:
        alg = r.get("algorithm", r.get("tag", "?"))
        parts = [f"    [{alg}]"]
        if "pearson_r"         in r: parts.append(f"Pearson r={r['pearson_r']:.4f}")
        if "spearman_rho"      in r: parts.append(f"Spearman ρ={r['spearman_rho']:.4f}")
        if "intra_inter_ratio" in r: parts.append(f"Intra/Inter={r['intra_inter_ratio']:.4f}")
        print("  ".join(parts))
    print("=" * 60)

def main():
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory.")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    d = load_results(results_dir)

    print_summary_table(d)

    print("\nGenerating plots …")
    plot_projections(d)
    plot_cluster_sizes(d)
    plot_silhouette(d)
    plot_cluster_metrics(d)
    plot_distance_distributions(d)
    plot_correlation_summary(d)
    plot_feature_importance(d)
    plot_embedding_stats(d)
    plot_embedding_correlation(d)

    print(f"\nAll plots saved to: {os.path.join(results_dir, 'plots')}/")


if __name__ == "__main__":
    main()