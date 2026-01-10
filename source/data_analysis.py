import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

save_dir = os.path.join("../data_analysis", dataset)
os.makedirs(save_dir, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

if dataset == 'private_au':
    df, dataset_name = load_raw_data("data/Solar_Energy_Generation_CK2.csv", target_col=target_column)
elif dataset == 'unisolar':
    df, dataset_name = load_unisolar_data("data/unisolar", target_site_key=unisolar_site_key)
else:
    raise ValueError(f"Unknown dataset: {dataset}")

##################################################################################################
# Config

features_for_hist = [target_column, 'AirTemperature', 'ghi', 'cloud_opacity', 'wind_speed_10m']
# features_for_hist = df.columns.tolist()

# features_to_pca = df.drop(columns=[target_column])  # exclude the target
features_to_pca = df[['ghi', 'SunAzimuth', 'precipitable_water', 'AirTemperature', 'dni']]
##################################################################################################

df = df.dropna()


print("\n--- Data Statistics ---")

print("\nData Info (Types):")
df.info()
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDescriptive Stats:\n{df.describe().T}")


print("\n--- Visualizations ---")

print("Plotting time-series... (this may take a moment)")
plt.figure(figsize=(14, 7))
df[target_column].plot(alpha=0.8, label=target_column)
df[target_column].resample('D').mean().plot(style='--', label='Daily Mean')
plt.title('Solar Generation Over Time')
plt.ylabel('Solar Generation')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, '1_timeseries_plot.png'))
print("Saved 1_timeseries_plot.png")

print("Plotting feature distributions...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(features_for_hist, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=50)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, '2_distribution_histograms.png'))
print("Saved 2_distribution_histograms.png")

print("Plotting boxplots for outliers...")
plt.figure(figsize=(15, 10))

scaler = StandardScaler()
df_scaled_for_plot = pd.DataFrame(scaler.fit_transform(df.drop(columns='SolarGeneration')),
                                  columns=df.columns.drop('SolarGeneration'))
sns.boxplot(data=df_scaled_for_plot, orient='h')
plt.title('Boxplots of Scaled Features (for Outlier Detection)')
plt.savefig(os.path.join(save_dir, '3_boxplots.png'))
print("Saved 3_boxplots.png")


print("\n--- Feature Analysis ---")

print("Calculating correlation matrix...")
df_numeric = df.select_dtypes(include=['float64', 'int64'])
corr_matrix = df_numeric.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Heatmap')
plt.savefig(os.path.join(save_dir, '4_correlation_heatmap.png'))
print("Saved heatmap")

print("Calculating feature importance...")
df_model = df_numeric.dropna()

X = df_model.drop(columns='SolarGeneration')
y = df_model['SolarGeneration']

if not X.empty and not y.empty:
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index)
    plt.title('Feature Importance (from Random Forest)')
    plt.savefig(os.path.join(save_dir, '5_feature_importance.png'))
    print("Saved 5_feature_importance.png")

    print(f"\nFeature Importances:\n{importances}")

else:
    print("Skipping feature importance: Not enough data after dropping NaNs.")

print("\n--- Chi-Square Test ---")
print("Note: Continuous variables are binned into quartiles for this test.")

df_binned = pd.DataFrame()
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    try:
        # ensure roughly equal number of samples in each bin
        df_binned[col] = pd.qcut(df[col], q=80, labels=False, duplicates='drop')
    except ValueError:
        df_binned[col] = df[col]

features = df_binned.columns
p_values = pd.DataFrame(index=features, columns=features, dtype=float)

for col1 in features:
    for col2 in features:
        if col1 == col2:
            p_values.loc[col1, col2] = np.nan  # Ignore diagonal
            continue

        contingency_table = pd.crosstab(df_binned[col1], df_binned[col2])

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        p_values.loc[col1, col2] = p

plt.figure(figsize=(12, 10))
sns.heatmap(1 - p_values, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1)
plt.title('Chi-Square Dependence Heatmap (1 - p_value)\nDarker Red = Strongly Dependent')
plt.savefig(os.path.join(save_dir, '6_chi_square_dependence_heatmap.png'))
plt.show()

print("\n--- Running PCA ---")

def run_pca(df):
    target = df['SolarGeneration']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_to_pca)

    pca = PCA(n_components=None)
    X_pca = pca.fit_transform(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cum_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(1, len(cum_variance) + 1), cum_variance, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.title('PCA Scree Plot')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, '7_pca_scree_plot.png'))
    plt.show()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap='viridis', alpha=0.5)
    plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} Variance)')
    plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} Variance)')
    plt.title('PCA: 2D Projection of Solar Dataset')
    plt.colorbar(scatter, label='Solar Generation (kW)')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, '8_pca_2d_projection.png'))
    plt.show()

    loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i + 1}' for i in range(len(features_to_pca.columns))],
                            index=features_to_pca.columns)
    print("\nTop Features contributing to PC1 (absolute weight):")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5))

run_pca(df)
