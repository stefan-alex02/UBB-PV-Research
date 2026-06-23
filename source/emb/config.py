"""
Pipeline configuration. Edit this file to change hyperparameters and technique choices.
"""
from datetime import datetime

EXP_NAME = "win6_latest_y_default_large_ae"

# Windowing
WINDOW_SIZE = 6   # k: number of consecutive rows per sample

# Windowing mode — how y (target) is associated with each window:
#   "next_y"   : window keeps ALL columns (incl. SolarGeneration); target y is the
#                SolarGeneration of the NEXT timestep. Last window dropped. (forecasting)
#   "latest_y" : window EXCLUDES SolarGeneration; target y is the SolarGeneration of
#                the LATEST timestep in the window. No window sacrificed.
WINDOWING_MODE = "latest_y"

# Name of the target column (solar generation)
TARGET_COL = "SolarGeneration"

# Normalization
# Options: "standard" (z-score), "minmax", "none"
NORMALIZATION = "standard"

# Embedding
# Options: "identity", "autoencoder", "univariate_fs", "random_forest_fs", "relieff"
EMBEDDING_METHOD = "autoencoder"

# Whether to standardize (z-score) the EMBEDDING before clustering/analysis.
# Autoencoder latent dims often have very different scales; standardizing prevents
# high-variance dims from dominating Euclidean distance. Recommended: True.
NORMALIZE_EMBEDDINGS = False

# Autoencoder settings (used when EMBEDDING_METHOD == "autoencoder") ---
AE_LATENT_DIM    = 32       # Embedding dimension
# Encoder hidden layer sizes (decoder mirrors this). Two options:
#   - explicit list, e.g. [256, 128, 64]  (used as-is)
#   - "auto"  → geometric funnel auto-derived from input dim & latent dim,
#               scaling automatically with WINDOW_SIZE. Recommended.
# AE_HIDDEN_DIMS   = "auto"
AE_HIDDEN_DIMS   = [256, 128, 64]
AE_AUTO_N_LAYERS = 2        # number of hidden layers when AE_HIDDEN_DIMS == "auto"
AE_EPOCHS        = 50
AE_BATCH_SIZE    = 64
AE_LEARNING_RATE = 1e-3
AE_DROPOUT       = 0.1      # Dropout rate in encoder/decoder layers
AE_VAL_SPLIT     = 0.15     # Fraction of training data held out for validation (0 = disabled)
AE_WEIGHT_DECAY  = 1e-5     # L2 regularization strength (Adam weight_decay). 0 = off.

# --- Feature selection settings (used for *_fs methods) ---
FS_N_FEATURES = 32          # Number of features to keep

# Random Forest FS
RF_N_ESTIMATORS = 100

# ReliefF
RELIEFF_N_NEIGHBORS = 10

# Clustering
# List any subset of: "kmeans", "dbscan", "agglomerative", "kmedoids", "gmm".
# Each selected method is computed and evaluated independently.
CLUSTERING_METHODS = ["kmeans", "agglomerative", "kmedoids", "gmm"]

# --- K-Means ---
KMEANS_K = 4
KMEANS_INIT = "k-means++"   # "k-means++" or "random"
KMEANS_MAX_ITER = 300

# --- DBSCAN ---
DBSCAN_EPS    = 5.
DBSCAN_MIN_SAMPLES = 5

# --- Agglomerative (hierarchical) ---
AGG_N_CLUSTERS = 4
AGG_LINKAGE    = "ward"   # "ward" (Euclidean only), "complete", "average", "single"
AGG_CONNECTIVITY = "knn"   # "none", "knn" (k-nearest neighbors), or "radius" (radius neighbors)
AGG_CONNECTIVITY_PARAM = 15   # k (int) for knn, radius (float) for radius neighbors

# --- K-Medoids (PAM) ---
KMEDOIDS_K        = 4
KMEDOIDS_METRIC   = "euclidean"   # any sklearn pairwise metric, e.g. "manhattan", "cosine"
KMEDOIDS_MAX_ITER = 300

# --- Gaussian Mixture Model ---
GMM_N_COMPONENTS   = 4
GMM_COVARIANCE_TYPE = "full"   # "full", "tied", "diag", "spherical"


# --- Distance metric for clustering analysis ---
# Options: "euclidean", "mahalanobis"
DISTANCE_METRIC = "euclidean"

# Analysis
# Options: "pearson", "spearman", "both"
CORRELATION_METHOD = "both"

# Number of pairs to sample for pairwise distance correlation (None = all, can be slow)
ANALYSIS_SAMPLE_PAIRS = 5000

# External (target-aware) analysis
# Discretization of y into classes for the Homogeneity metric.
# Class 0 = "low" : y < LOW_CLASS_FRACTION * y_max  (covers night + small error margin)
# Classes 1..3    = remaining range [LOW_CLASS_FRACTION*y_max, y_max] split equally.
N_Y_CLASSES        = 4
LOW_CLASS_FRACTION = 0.05   # 5% of max y is the upper bound of the "low" class

# External intra/inter correlation: cap pairs per cluster (or per cluster-pair) for speed.
EXT_MAX_PAIRS_PER_GROUP = 20000   # None = use all pairs (can be very slow for big clusters)

# === Supervised forecasting probe =============================================================
# A frozen probe: train a regressor to predict y from the (already-computed)
# representation, with time-series cross-validation. Answers "does this embedding
# carry forecast-relevant signal?" and provides interpretable feature importances.
RUN_SUPERVISED = False

# Which model to probe with: "xgboost", "random_forest", "mlp".
SUPERVISED_MODEL = "xgboost"

# Which representations to probe:
#   "embedding" → on Z (gives the embedding-usefulness score; latent dims not interpretable)
#   "raw"       → on the raw windowed features (interpretable importances mapped to columns)
SUPERVISED_ON = ["embedding", "raw"]

# Time-series cross-validation folds (expanding-window forward chaining; no shuffling).
CV_N_SPLITS = 5

# Explainability
RUN_SHAP = True          # SHAP attributions (TreeExplainer for tree models; slow KernelSHAP for MLP)
SHAP_MAX_SAMPLES = 500   # cap samples passed to SHAP for speed

# --- XGBoost params ---
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH    = 6
XGB_LEARNING_RATE= 0.1
XGB_SUBSAMPLE    = 0.9

# --- Random Forest (supervised) params ---
SUP_RF_N_ESTIMATORS = 300
SUP_RF_MAX_DEPTH    = None

# --- MLP params ---
MLP_HIDDEN_LAYERS = [128, 64]
MLP_ALPHA         = 1e-4   # L2 penalty
MLP_LEARNING_RATE = 1e-3
MLP_MAX_ITER      = 300

# ===============================================================================================

#  Misc
RANDOM_SEED = 42
DEVICE = "cuda"   # "cuda" or "cpu"

# Dataset
DATASET_PATH = "../../data/private_au/Solar_Energy_Generation_CK2.csv"
TIMESTAMP_COL = "Timestamp"          # Column to drop before processing
_now = datetime.now() # underscore to avoid saving it in the results json
RESULTS_DIR   = f"results/{EMBEDDING_METHOD}/{_now.strftime('%Y-%m-%d_%H-%M-%S')}_{EXP_NAME}"
PLOT = True