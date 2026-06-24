"""
Visualise the outputs of the PV embedding pipeline.

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

# Helpers

def load_results(results_dir: str) -> dict:
    data = {"dir": results_dir}

    path = lambda f: os.path.join(results_dir, f)

    data["Z"] = np.load(path("embeddings.npy"))

    data["labels"] = {}
    label_files = sorted(f for f in os.listdir(results_dir)
                         if f.startswith("labels_") and f.endswith(".npy"))
    for fname in label_files:
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

    # New target-aware artifacts (optional)
    def _opt_npy(name):
        p = path(name)
        return np.load(p) if os.path.exists(p) else None

    def _opt_json(name):
        p = path(name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    data["y"]            = _opt_npy("y_targets.npy")
    data["y_classes"]    = _opt_npy("y_classes.npy")
    data["class_info"]   = _opt_json("y_class_info.json")
    data["external"]     = _opt_json("external_correlation_analysis.json")
    data["ae_history"]   = _opt_json("ae_training_history.json")
    data["supervised"]   = _opt_json("supervised_results.json")

    return data


def out(results_dir: str, name: str) -> str:
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return os.path.join(plots_dir, name)


def label_colors(labels, cmap=COLORMAP):
    unique = sorted(set(labels))
    cm = plt.get_cmap(cmap, max(len(unique), 2))
    return {lbl: ("lightgrey" if lbl == -1 else cm(i)) for i, lbl in enumerate(unique)}


# 1. PCA + t-SNE 2-D projections coloured by cluster

def plot_projections(d: dict):
    Z = d["Z"]
    labels_dict = d["labels"]
    cfg = d["config"]
    emb_method = cfg.get("EMBEDDING_METHOD", "")

    # Compute PCA (fast) and t-SNE (slower but better topology)
    pca2 = PCA(n_components=2, random_state=42).fit_transform(Z)
    print("  Computing t-SNE ...")
    tsne2 = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=500).fit_transform(Z)

    for alg, labels in labels_dict.items():
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            f"2-D Projections for embedding: {emb_method}, clustering: {alg.upper()}",
            fontsize=12, fontweight="bold"
        )

        for ax, proj, title in zip(axes, [pca2, tsne2], ["PCA", "t-SNE"]):
            cmap = label_colors(labels)
            colors = [cmap[l] for l in labels]
            ax.scatter(proj[:, 0], proj[:, 1], c=colors, s=8, alpha=0.6, linewidths=0)
            ax.set_title(title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            legend_els = [
                Patch(color=cmap[l], label="Noise" if l == -1 else f"C{l}")
                for l in sorted(set(labels))
            ]
            ax.legend(handles=legend_els, fontsize=7, markerscale=1.5,
                      ncol=max(1, len(legend_els) // 10), loc="best")

        plt.tight_layout()
        fpath = out(d["dir"], f"projections_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 2. Cluster size distribution bar chart

def _alg_for_summary_index(d: dict, idx: int) -> str:
    """Map a cluster_summary_<idx> to its algorithm name via cluster_metrics row order
    (the authoritative order from save_results), not filesystem listing order."""
    alg_order = d["cluster_metrics"]["algorithm"].tolist()
    return alg_order[idx] if idx < len(alg_order) else f"clustering_{idx}"


def plot_cluster_sizes(d: dict):
    """One standalone cluster-size figure per clustering method."""
    summaries = d["summaries"]
    if not summaries:
        return

    for idx_str in sorted(summaries.keys(), key=lambda s: int(s)):
        df = summaries[idx_str]
        alg = _alg_for_summary_index(d, int(idx_str))
        cmap = plt.get_cmap(COLORMAP, max(len(df), 2))
        n_clusters = len(df)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if n_clusters > 15:
            top_df = df.nlargest(20, "size")
            colors = ["lightgrey" if r["cluster"] == -1 else cmap(j % 10)
                      for j, (_, r) in enumerate(top_df.iterrows())]
            bars = ax.barh([str(c) for c in top_df["cluster"]], top_df["size"],
                    color=colors, edgecolor="white")
            ax.invert_yaxis()
            ax.set_xlabel("Number of samples")
            ax.set_ylabel("Cluster")
            # Ensure there's a little space on the right for numeric labels
            xmax = top_df["size"].max() if len(top_df) else 1
            ax.set_xlim(0, xmax * 1.10)
            # Label the bars with their counts (numbers placed to the right)
            ax.bar_label(bars, fmt="%d", padding=4, fontsize=8)
            n_noise = int(df.loc[df["cluster"] == -1, "size"].sum()) if -1 in df["cluster"].values else 0
            ax.set_title(f"Cluster sizes (top 20) ({alg.upper()})\n"
                         f"Total clusters: {n_clusters}  |  Noise: {n_noise}", fontweight="bold")
        else:
            colors = ["lightgrey" if r["cluster"] == -1 else cmap(j % 10)
                      for j, (_, r) in enumerate(df.iterrows())]
            bars = ax.bar([str(c) for c in df["cluster"]], df["size"],
                          color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%d", fontsize=8)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Number of samples")
            ax.set_title(f"Cluster sizes ({alg.upper()})", fontweight="bold")

        plt.tight_layout()
        fpath = out(d["dir"], f"cluster_sizes_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 3. Silhouette plot per sample

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
        ax.set_title(f"Silhouette analysis ({alg.upper()})", fontweight="bold")
        ax.legend()

        plt.tight_layout()
        fpath = out(d["dir"], f"silhouette_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 4. Cluster quality metrics bar chart

def plot_cluster_metrics(d: dict):
    """One standalone metrics figure per clustering method."""
    df = d["cluster_metrics"]
    metric_cols = [c for c in ["silhouette", "davies_bouldin", "calinski_harabasz"]
                   if c in df.columns]
    if not metric_cols:
        return

    notes = {
        "silhouette":        "higher is better (range -1 to 1)",
        "davies_bouldin":    "lower is better",
        "calinski_harabasz": "higher is better",
    }

    for _, row in df.iterrows():
        alg = row["algorithm"]
        fig, axes = plt.subplots(1, len(metric_cols), figsize=(4 * len(metric_cols), 4))
        if len(metric_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, metric_cols):
            val = float(row[col])
            bars = ax.bar([alg], [val], color="#4C72B0", edgecolor="white", width=0.5)
            ax.bar_label(bars, fmt="%.3f", fontsize=10)
            ax.set_title(f"{col.replace('_', ' ').title()}\n{notes.get(col,'')}",
                         fontweight="bold", fontsize=10)
            ax.set_ylabel(col)
        fig.suptitle(f"Cluster quality metrics ({alg.upper()})", fontweight="bold")
        plt.tight_layout()
        fpath = out(d["dir"], f"cluster_metrics_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 5. Intra vs inter cluster distance distributions

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
        ax.set_title(f"Distance distribution ({alg.upper()})", fontweight="bold")
        ax.legend()

        plt.tight_layout()
        fpath = out(d["dir"], f"distance_dist_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 6. Correlation scatter: embedding distance vs same-cluster indicator

def plot_correlation_summary(d: dict):
    """One standalone internal-correlation figure per clustering method."""
    corr_list = d["correlation"]
    if not corr_list:
        return

    notes = {
        "Pearson r":         "more negative means distances reflect clusters better",
        "Spearman rho":        "more negative means distances reflect clusters better",
        "Intra/Inter ratio": "lower means more compact and separated clusters",
    }

    for r in corr_list:
        alg = r.get("algorithm", r.get("tag", "?"))
        vals = {}
        if "pearson_r"        in r: vals["Pearson r"]          = r["pearson_r"]
        if "spearman_rho"     in r: vals["Spearman rho"]         = r["spearman_rho"]
        if "intra_inter_ratio"in r: vals["Intra/Inter ratio"]  = r["intra_inter_ratio"]
        if not vals:
            continue

        cols = list(vals.keys())
        fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            bars = ax.bar([alg], [float(vals[col])], color="#4C72B0",
                          edgecolor="white", width=0.5)
            ax.bar_label(bars, fmt="%.4f", fontsize=10)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{col}\n{notes.get(col,'')}", fontweight="bold", fontsize=9)
        fig.suptitle(f"Internal correlation ({alg.upper()})",
                     fontweight="bold")
        plt.tight_layout()
        fpath = out(d["dir"], f"correlation_summary_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 7. Feature importance (for FS-based embedding methods)

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
        title = "Univariate FS (F-scores)"
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
    ax.set_title(f"{title} (top {top_n})", fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "feature_importance.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# 8. Embedding dimension statistics (mean +/- std per dimension)

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


# 9. Embedding correlation heatmap

def plot_embedding_correlation(d: dict):
    Z = d["Z"]
    if Z.shape[1] > 64:
        print(f"  [Skip] Embedding dim {Z.shape[1]} > 64 - skipping correlation heatmap.")
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


# 10. Autoencoder training history (train vs val loss)

def plot_training_history(d: dict):
    hist = d.get("ae_history")
    if not hist or not hist.get("epoch"):
        return  # not an autoencoder run

    epochs = hist["epoch"]
    train  = hist["train_loss"]
    val    = hist.get("val_loss", [])
    has_val = val and not all(v != v for v in val)  # not all NaN

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train, label="Train loss", color="#4C72B0", linewidth=1.8)
    if has_val:
        ax.plot(epochs, val, label="Validation loss", color="#DD8452", linewidth=1.8)
        # Mark best epoch by val loss
        val_arr = np.array(val, dtype=float)
        best = int(np.nanargmin(val_arr))
        ax.axvline(epochs[best], color="grey", linestyle="--", linewidth=1,
                   label=f"Best val @ epoch {epochs[best]} ({val_arr[best]:.4f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE reconstruction loss")
    ax.set_title("Autoencoder training history", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    fpath = out(d["dir"], "training_history.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# 11. External intra/inter correlation (target-aware)

def plot_external_correlation(d: dict):
    """One standalone external-correlation figure per clustering method."""
    ext = d.get("external")
    if not ext:
        return

    for r in ext:
        alg = r.get("algorithm", r.get("tag", "?"))
        groups = ["Intra", "Inter"]
        pearson  = [r.get("intra_corr_pearson_mean"),  r.get("inter_corr_pearson_mean")]
        spearman = [r.get("intra_corr_spearman_mean"), r.get("inter_corr_spearman_mean")]

        x = np.arange(len(groups))
        w = 0.35
        fig, ax = plt.subplots(figsize=(7, 4.5))
        b1 = ax.bar(x - w/2, pearson,  w, label="Pearson",  color="#4C72B0", edgecolor="white")
        b2 = ax.bar(x + w/2, spearman, w, label="Spearman", color="#DD8452", edgecolor="white")
        ax.bar_label(b1, fmt="%.3f", fontsize=8)
        ax.bar_label(b2, fmt="%.3f", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}-cluster" for g in groups])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Correlation")
        ax.set_title(f"External correlation ({alg.upper()})\n"
                     f"higher = geometry tracks target",
                     fontweight="bold", fontsize=10)
        ax.legend()

        plt.tight_layout()
        fpath = out(d["dir"], f"external_correlation_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 12. Homogeneity score per algorithm

def plot_homogeneity(d: dict):
    """One standalone homogeneity figure per clustering method."""
    df = d["cluster_metrics"]
    if "homogeneity" not in df.columns:
        return

    for _, row in df.iterrows():
        alg = row["algorithm"]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        bars = ax.bar([alg], [float(row["homogeneity"])], color="#4C72B0",
                      edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.3f", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Homogeneity")
        ax.set_title(f"Homogeneity vs y-classes ({alg.upper()})\nhigher is better, 0-1",
                     fontweight="bold", fontsize=10)
        plt.tight_layout()
        fpath = out(d["dir"], f"homogeneity_{alg}.png")
        plt.savefig(fpath, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fpath}")


# 13. Target distribution with class boundaries

def plot_target_distribution(d: dict):
    y = d.get("y")
    info = d.get("class_info")
    if y is None or info is None:
        return

    edges = info["bin_edges"]
    y_max = info["y_max"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(y, bins=80, color="#4C72B0", alpha=0.8, edgecolor="white")
    cmap = plt.get_cmap("tab10")
    for c, cr in enumerate(info["class_ranges"]):
        ax.axvspan(cr["low"], cr["high"], color=cmap(c), alpha=0.12)
        mid = (cr["low"] + cr["high"]) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.92, f"C{c}\nn={cr['count']}",
                ha="center", va="top", fontsize=8, color=cmap(c), fontweight="bold")
    for e in edges[1:-1]:
        ax.axvline(e, color="red", linestyle="--", linewidth=1)

    ax.set_xlabel("Solar Generation (raw units)")
    ax.set_ylabel("Count")
    ax.set_title(f"Target distribution with class boundaries  (y_max = {y_max:.1f})",
                 fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "target_distribution.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# 14. Supervised probe: CV scores

def plot_supervised_scores(d: dict):
    sup = d.get("supervised")
    if not sup or "cv_summary" not in sup:
        return

    model_name = sup.get("model", "model")
    summary = sup["cv_summary"]
    folds = sup.get("cv_folds", [])
    metrics = [("r2", "R2 (higher better)"), ("mae", "MAE (lower better)"),
               ("rmse", "RMSE (lower better)"), ("nrmse", "NRMSE (lower better)")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, metrics):
        mean = summary.get(f"{key}_mean", float("nan"))
        ci   = summary.get(f"{key}_ci95", summary.get(f"{key}_std", 0.0))
        # Bar shows the CV mean with a 95% confidence interval; points show each fold.
        bar = ax.bar(["CV mean"], [mean], yerr=[ci], capsize=6,
                     color="#4C72B0", edgecolor="white", width=0.5)
        ax.bar_label(bar, fmt="%.3f", fontsize=9, padding=3)
        if folds:
            fold_vals = [f[key] for f in folds]
            ax.scatter(np.zeros(len(fold_vals)), fold_vals, color="#DD8452",
                       zorder=3, s=25, label="per fold")
            ax.legend(fontsize=7)
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel(key)
    fig.suptitle(f"Supervised forecast probe ({model_name}) on embeddings, time series CV "
                 f"(error bars: 95% CI)", fontweight="bold")

    plt.tight_layout()
    fpath = out(d["dir"], "supervised_scores.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def _barh_top(values_by_name, top_n, title, xlabel, fname, d, color="#4C72B0"):
    """Generic horizontal bar chart of the top_n entries of a name to value dict."""
    if not values_by_name:
        return
    items = sorted(values_by_name.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items]
    vals  = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.4)))
    ax.barh(range(len(names)), vals[::-1], color=color, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold", fontsize=10)

    plt.tight_layout()
    fpath = out(d["dir"], fname)
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_supervised_importance(d: dict):
    """Top original columns (averaged over time steps) by permutation importance and,
    separately, by SHAP. Also the raw time step specific features for each."""
    sup = d.get("supervised")
    if not sup:
        return
    model_name = sup.get("model", "model")
    top_n = sup.get("top_n", 15)
    names = sup.get("feature_names", [])

    # Permutation importance, aggregated to original columns
    perm_by_col = sup.get("importances_by_column")
    if perm_by_col:
        _barh_top(perm_by_col, top_n,
                  f"Top columns by permutation importance, {model_name}\n"
                  f"(averaged over window time steps)",
                  "Permutation importance (drop in R2)",
                  "supervised_importance_columns.png", d)

    # Permutation importance, raw time step specific features
    perm = sup.get("importances")
    if perm and names and len(perm) == len(names):
        raw_map = {names[i]: float(perm[i]) for i in range(len(names))}
        _barh_top(raw_map, top_n,
                  f"Top raw time step features by permutation importance, {model_name}",
                  "Permutation importance (drop in R2)",
                  "supervised_importance_raw_timesteps.png", d)

    # SHAP, aggregated to original columns
    shap_by_col = sup.get("shap_by_column")
    if shap_by_col:
        _barh_top(shap_by_col, top_n,
                  f"Top columns by SHAP, {model_name}\n"
                  f"(averaged over window time steps)",
                  "Mean absolute SHAP value",
                  "supervised_shap_columns.png", d, color="#55A868")

    # SHAP, raw time step specific features
    shap_raw = sup.get("shap_mean_abs")
    if shap_raw and names and len(shap_raw) == len(names):
        raw_map = {names[i]: float(shap_raw[i]) for i in range(len(names))}
        _barh_top(raw_map, top_n,
                  f"Top raw time step features by SHAP, {model_name}",
                  "Mean absolute SHAP value",
                  "supervised_shap_raw_timesteps.png", d, color="#55A868")


def plot_supervised_training_curves(d: dict):
    """Per fold training (and internal validation) curves, where the model produces
    them: xgboost RMSE per boosting round, or MLP loss per epoch. Random forest has
    no iterative curve and is skipped."""
    sup = d.get("supervised")
    if not sup:
        return
    histories = sup.get("cv_fold_histories") or []
    if not histories:
        return

    model_name = sup.get("model", "model")
    metric_name = histories[0].get("curve_metric", "loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10", max(len(histories), 2))
    for i, h in enumerate(histories):
        it = h.get("iterations", [])
        tr = h.get("train_curve")
        va = h.get("val_curve")
        color = cmap(i)
        fold = h.get("fold", i + 1)
        if tr:
            ax.plot(it, tr, color=color, linewidth=1.6, label=f"fold {fold} train")
        if va and h.get("val_curve_metric") != "r2":
            # same scale as train (e.g. xgboost rmse): plot dashed
            ax.plot(it, va, color=color, linewidth=1.4, linestyle="--",
                    label=f"fold {fold} val")

    ax.set_xlabel("iteration (boosting round or epoch)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Per fold training curves, {model_name}\n"
                 f"(internal time ordered validation tail; dashed = validation)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fpath = out(d["dir"], "supervised_training_curves.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


def plot_supervised_fold_trend(d: dict):
    """How each metric evolves across the time ordered CV folds. Useful to see whether
    forecast quality improves as the training window grows."""
    sup = d.get("supervised")
    if not sup:
        return
    folds = sup.get("cv_folds") or []
    if not folds:
        return
    model_name = sup.get("model", "model")

    fold_ids = [f["fold"] for f in folds]
    metrics = [("r2", "R2"), ("nrmse", "NRMSE")]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()
    l1 = ax1.plot(fold_ids, [f["r2"] for f in folds], "o-", color="#4C72B0", label="R2")
    l2 = ax2.plot(fold_ids, [f["nrmse"] for f in folds], "s--", color="#DD8452", label="NRMSE")
    ax1.set_xlabel("CV fold (chronological, training window grows)")
    ax1.set_ylabel("R2", color="#4C72B0")
    ax2.set_ylabel("NRMSE", color="#DD8452")
    ax1.set_xticks(fold_ids)
    ax1.set_title(f"Forecast quality across CV folds, {model_name}", fontweight="bold")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="best")

    plt.tight_layout()
    fpath = out(d["dir"], "supervised_fold_trend.png")
    plt.savefig(fpath, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fpath}")


# 10. Summary statistics table

def print_summary_table(d: dict):
    cfg = d["config"]
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  PIPELINE SUMMARY")
    lines.append("=" * 60)
    lines.append(f"  Embedding method  : {cfg.get('EMBEDDING_METHOD')}")
    lines.append(f"  Window size (k)   : {cfg.get('WINDOW_SIZE')}")
    lines.append(f"  Normalization     : {cfg.get('NORMALIZATION')}")
    lines.append(f"  Distance metric   : {cfg.get('DISTANCE_METRIC')}")
    lines.append(f"  Embedding shape   : {d['Z'].shape}")
    lines.append("")
    lines.append("  Clustering metrics:")
    lines.append(d["cluster_metrics"].to_string(index=False))
    lines.append("")
    lines.append("  Correlation analysis (internal - geometry vs cluster):")

    for r in d["correlation"]:
        alg = r.get("algorithm", r.get("tag", "?"))
        parts = [f"    [{alg}]"]
        if "pearson_r" in r:
            parts.append(f"Pearson r={r['pearson_r']:.4f}")
        if "spearman_rho" in r:
            parts.append(f"Spearman rho={r['spearman_rho']:.4f}")
        if "intra_inter_ratio" in r:
            parts.append(f"Intra/Inter={r['intra_inter_ratio']:.4f}")
        lines.append("  ".join(parts))

    if d.get("external"):
        lines.append("")
        lines.append("  External correlation (target-aware - embedding dist vs |dy|):")
        for r in d["external"]:
            alg = r.get("algorithm", r.get("tag", "?"))
            lines.append(
                f"    [{alg}]  Intra(P)={r.get('intra_corr_pearson_mean', float('nan')):.4f}  "
                f"Inter(P)={r.get('inter_corr_pearson_mean', float('nan')):.4f}  "
                f"Intra(S)={r.get('intra_corr_spearman_mean', float('nan')):.4f}  "
                f"Inter(S)={r.get('inter_corr_spearman_mean', float('nan')):.4f}"
            )

    if "homogeneity" in d["cluster_metrics"].columns:
        lines.append("")
        lines.append("  Homogeneity (vs y-classes):")
        for _, row in d["cluster_metrics"].iterrows():
            lines.append(f"    [{row['algorithm']}]  Homogeneity={row['homogeneity']:.4f}")

    sup = d.get("supervised")
    if sup and sup.get("cv_summary"):
        s = sup["cv_summary"]
        lines.append("")
        lines.append(f"  Supervised forecast probe (model={sup.get('model','?')}, "
                     f"trained on embeddings, time series CV):")
        def _fmt(key, label):
            mean = s.get(f"{key}_mean", float('nan'))
            ci   = s.get(f"{key}_ci95", 0.0)
            return f"{label}={mean:.4f} +/- {ci:.4f}"
        lines.append("    " + "   ".join([
            _fmt("r2", "R2"), _fmt("mae", "MAE"),
            _fmt("rmse", "RMSE"), _fmt("nrmse", "NRMSE"),
        ]) + "   (means +/- 95% CI)")

        top_imp = sup.get("top_importance_columns")
        if top_imp:
            names = ", ".join(f"{e['name']}({e['value']:.3f})" for e in top_imp[:5])
            lines.append(f"    Top columns by permutation importance: {names}")
        top_shap = sup.get("top_shap_columns")
        if top_shap:
            names = ", ".join(f"{e['name']}({e['value']:.3f})" for e in top_shap[:5])
            lines.append(f"    Top columns by SHAP: {names}")

    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)

    summary_path = os.path.join(d["dir"], "pipeline_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    print(f"  Saved: {summary_path}")


def plot(results_dir: str = None, generate_plots: bool = True):
    if results_dir is None:
        results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory.")
        sys.exit(1)

    print(f"Loading results from: {results_dir}")
    d = load_results(results_dir)

    print_summary_table(d)

    if generate_plots:
        print("\nGenerating plots ...")
        plot_projections(d)
        plot_cluster_sizes(d)
        plot_silhouette(d)
        plot_cluster_metrics(d)
        plot_distance_distributions(d)
        plot_correlation_summary(d)
        plot_feature_importance(d)
        plot_embedding_stats(d)
        plot_embedding_correlation(d)
        plot_training_history(d)
        plot_external_correlation(d)
        plot_homogeneity(d)
        plot_target_distribution(d)
        plot_supervised_scores(d)
        plot_supervised_importance(d)
        plot_supervised_training_curves(d)
        plot_supervised_fold_trend(d)

    print(f"\nAll plots saved to: {os.path.join(results_dir, 'plots')}/")


# Main

def main():
    plot(results_dir)


if __name__ == "__main__":
    plot()
