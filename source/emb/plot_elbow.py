import os
from typing import Sequence, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def find_elbow_index(ks: Sequence[float], inertias: Sequence[float]) -> int:
    """
    Return the index of the elbow (knee) using the max perpendicular distance
    from the line connecting the first and last (k, inertia) points.
    """
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    if x.size < 3:
        # Fallback for very short series: pick the largest drop
        diffs = -np.diff(y)  # larger drops -> larger positive
        return int(np.argmax(diffs)) if diffs.size > 0 else 0

    # endpoints
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    # line coefficients A*x + B*y + C = 0
    A = y2 - y1
    B = -(x2 - x1)
    C = x2 * y1 - y2 * x1
    denom = np.hypot(A, B)
    if denom == 0:
        return int(np.argmax(y))  # degenerate fallback

    d = np.abs(A * x + B * y + C) / denom
    d[0] = 0.0
    d[-1] = 0.0
    return int(np.argmax(d))


def plot_elbow(
    ks: Sequence[int],
    inertias: Sequence[float],
    out_path: Optional[str] = None,
    annotate_points: bool = False,
    figsize: Tuple[int, int] = (7, 4),
    dpi: int = 150,
) -> Tuple[int, int]:
    """
    Plot inertia vs k, detect and mark elbow.

    Args:
      ks: sequence of k values (ints)
      inertias: sequence of inertia values (floats) matching ks
      out_path: if provided, save PNG to this path; otherwise show plot
      annotate_points: if True annotate every point with its inertia value
      figsize/dpi: plot size and resolution

    Returns:
      (elbow_index_in_sorted_arrays, elbow_k)
    """
    ks_arr = np.asarray(ks, dtype=float)
    inertias_arr = np.asarray(inertias, dtype=float)
    if ks_arr.shape != inertias_arr.shape:
        raise ValueError("ks and inertias must have the same length")

    # Sort by k to ensure increasing order
    order = np.argsort(ks_arr)
    ks_s = ks_arr[order]
    inertias_s = inertias_arr[order]

    elbow_idx = find_elbow_index(ks_s, inertias_s)
    elbow_k = int(ks_s[elbow_idx])
    elbow_val = float(inertias_s[elbow_idx])

    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(ks_s, inertias_s, marker="o", linestyle="-", color="#1f77b4")
    plt.scatter([elbow_k], [elbow_val], color="red", zorder=10, s=64, label=f"Elbow: k={elbow_k}")
    plt.axvline(elbow_k, color="red", linestyle="--", linewidth=1)

    if annotate_points:
        for x, y in zip(ks_s, inertias_s):
            plt.text(x, y, f"{y:.0f}", fontsize=8, ha="center", va="bottom", alpha=0.85)

    # Annotate the elbow with a small textbox + arrow
    plt.annotate(
        f"k={elbow_k}\nInertia={elbow_val:.0f}",
        xy=(elbow_k, elbow_val),
        xytext=(0.98, 0.95),
        textcoords="axes fraction",
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="red"),
    )

    plt.xlabel("k (n_clusters)")
    plt.ylabel("Inertia (sum of squared distances)")
    plt.title("KMeans Elbow Method")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        print(f"Saved elbow plot to: {out_path}")
    else:
        plt.show()

    # Return elbow index relative to the sorted arrays and the elbow k
    return elbow_idx, elbow_k


def main():

    ks = [2, 3, 4, 5, 6, 7, 8]
    inertias = [1200, 900, 700, 650, 630, 620, 615]
    idx, elbow_k = plot_elbow(ks, inertias, out_path="k_elbow.png", annotate_points=True)
    print("Elbow at k:", elbow_k, "index:", idx)


if __name__ == "__main__":
    main()
