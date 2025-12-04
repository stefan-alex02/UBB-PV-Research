import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a professional style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 150})


def load_experiment_data(experiment_dir):
    """
    Loads the three expected CSV files from the experiment directory.
    Returns None if a file is missing.
    """
    loss_path = os.path.join(experiment_dir, 'training_losses.csv')
    fold_path = os.path.join(experiment_dir, 'fold_metrics.csv')
    summary_path = os.path.join(experiment_dir, 'final_summary.csv')

    losses = pd.read_csv(loss_path) if os.path.exists(loss_path) else None
    folds = pd.read_csv(fold_path) if os.path.exists(fold_path) else None
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    if losses is None and folds is None:
        print(f"Warning: No valid CSVs found in {experiment_dir}")

    return losses, folds, summary


def plot_training_losses(losses_df, output_dir, show=False):
    """
    Creates two side-by-side plots: Training & Validation Loss.
    """
    if losses_df is None:
        print("Skipping Loss Plot: No loss data found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    n_folds = losses_df['Fold'].nunique()
    palette = sns.color_palette("viridis", n_folds)

    # Plot 1: Training Loss
    sns.lineplot(data=losses_df, x='Epoch', y='Train_Loss', hue='Fold',
                 palette=palette, ax=axes[0], legend=False)
    axes[0].set_title("Training Loss per Fold")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_xlabel("Epoch")

    # Plot 2: Validation Loss
    sns.lineplot(data=losses_df, x='Epoch', y='Valid_Loss', hue='Fold',
                 palette=palette, ax=axes[1])
    axes[1].set_title("Validation Loss per Fold")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(title='Fold', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'plot_loss_curves.png')
    plt.savefig(save_path)
    print(f"Saved Loss Plot: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_metrics_distributions(folds_df, output_dir, show=False):
    """
    Creates Horizontal Boxplots stacked vertically for key metrics.
    This visualizes the spread/stability of the model across folds.
    """
    if folds_df is None:
        return

    metrics_to_plot = ['MAE', 'RMSE', 'R2', 'MAPE']
    valid_metrics = [m for m in metrics_to_plot if m in folds_df.columns]

    if not valid_metrics:
        return

    # Create a vertical stack of subplots (1 column, N rows)
    fig, axes = plt.subplots(len(valid_metrics), 1, figsize=(10, 2.5 * len(valid_metrics)))

    if len(valid_metrics) == 1: axes = [axes]  # Handle single plot case

    for i, metric in enumerate(valid_metrics):
        ax = axes[i]

        # Horizontal Boxplot
        sns.boxplot(data=folds_df, x=metric, orient='h', ax=ax, color='lightblue', width=0.5)
        # Overlay actual data points (strip plot) so you can see the 5 folds
        sns.stripplot(data=folds_df, x=metric, orient='h', ax=ax, color='darkblue', size=6, alpha=0.8)

        ax.set_title(f"{metric} Distribution across Folds", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{metric} Value")
        ax.set_ylabel("")  # Remove y-label since it's just "0"

        # Grid helps reading values
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'plot_metrics_boxplot.png')
    plt.savefig(save_path)
    print(f"Saved Metrics Plot: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_summary_bar(summary_df, output_dir, show=False):
    """
    Creates a bar chart with error bars representing the 95% Confidence Interval.
    """
    if summary_df is None:
        return

    # Filter out metrics that have different scales
    error_metrics = summary_df[summary_df['Metric'].isin(['MAE', 'RMSE'])]

    if error_metrics.empty:
        return

    plt.figure(figsize=(8, 6))

    bars = plt.bar(error_metrics['Metric'], error_metrics['Mean'],
                   yerr=error_metrics['95% CI'], capsize=10,
                   color=['#1f77b4', '#ff7f0e'], alpha=0.8)

    plt.title("Model Performance (Mean ± 95% CI)")
    plt.ylabel("Error (kW)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'plot_summary_errors.png')
    plt.savefig(save_path)
    print(f"Saved Summary Plot: {save_path}")

    if show:
        plt.show()
    plt.close()

dir = "results/Solar_Energy_Generation_CK2/ANN/20251205_003857"
show_plots = True

print(f"--- Analyzing: {dir} ---")
losses, folds, summary = load_experiment_data(dir)

plot_training_losses(losses, dir, show=show_plots)
plot_metrics_distributions(folds, dir, show=show_plots)
plot_summary_bar(summary, dir, show=show_plots)
print("Analysis Complete.\n")
