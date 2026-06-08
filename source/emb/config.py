"""
Pipeline configuration. Edit this file to change hyperparameters and technique choices.
"""

# Dataset
DATASET_PATH = "../../data/private_au/Solar_Energy_Generation_CK2.csv"
TIMESTAMP_COL = "Timestamp"          # Column to drop before processing
RESULTS_DIR   = "results/random_forest_fs/run1"

# Windowing
WINDOW_SIZE = 6   # k+1: number of consecutive rows per sample

# Normalization
# Options: "standard", "minmax", "none"
NORMALIZATION = "standard"

# Embedding
# Options: "identity", "autoencoder", "univariate_fs", "random_forest_fs", "relieff"
EMBEDDING_METHOD = "random_forest_fs"

# Autoencoder settings (used when EMBEDDING_METHOD == "autoencoder") ---
AE_LATENT_DIM    = 32       # Embedding dimension
AE_HIDDEN_DIMS   = [256, 128, 64]  # Encoder hidden layer sizes (decoder mirrors this)
AE_EPOCHS        = 50
AE_BATCH_SIZE    = 64
AE_LEARNING_RATE = 1e-3
AE_DROPOUT       = 0.1      # Dropout rate in encoder/decoder layers

# --- Feature selection settings (used for *_fs methods) ---
FS_N_FEATURES = 32          # Number of features to keep

# Random Forest FS
RF_N_ESTIMATORS = 100

# ReliefF
RELIEFF_N_NEIGHBORS = 10

# Clustering
# Options: "kmeans", "dbscan", "both"
CLUSTERING_METHOD = "both"

# --- K-Means ---
KMEANS_K = 8
KMEANS_INIT = "k-means++"   # "k-means++" or "random"
KMEANS_MAX_ITER = 300

# --- DBSCAN ---
DBSCAN_EPS    = 3.
DBSCAN_MIN_SAMPLES = 5

# --- Distance metric for clustering analysis ---
# Options: "euclidean", "mahalanobis"
DISTANCE_METRIC = "euclidean"

# Analysis
# Options: "pearson", "spearman", "both"
CORRELATION_METHOD = "both"

# Number of pairs to sample for pairwise distance correlation (None = all, can be slow)
ANALYSIS_SAMPLE_PAIRS = 5000

# Misc
RANDOM_SEED = 42
DEVICE = "cpu"   # "cuda" or "cpu"
