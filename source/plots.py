import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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


def plot_predictions_scatter(experiment_dir, show=False):
    """
    Plots Predicted vs Actual values.
    Includes a 45-degree line (perfect prediction).
    """
    pred_path = os.path.join(experiment_dir, 'predictions.csv')
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)

    # Create plot
    plt.figure(figsize=(8, 8))

    # 1. Plot the data points
    # Using alpha (transparency) helps see density if points overlap
    sns.scatterplot(data=df, x='Actual', y='Predicted',
                    hue='Fold', palette='viridis', alpha=0.3, s=15)

    # 2. Add the "Perfect Prediction" diagonal line
    min_val = min(df['Actual'].min(), df['Predicted'].min())
    max_val = max(df['Actual'].max(), df['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val],
             color='red', linestyle='--', linewidth=2, label='Perfect Fit')

    plt.title("Predicted vs Actual Solar Generation")
    plt.xlabel("Actual Power (kW)")
    plt.ylabel("Predicted Power (kW)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save
    save_path = os.path.join(experiment_dir, 'plot_scatter_pred_vs_actual.png')
    plt.savefig(save_path)
    print(f"Saved Scatter Plot: {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_predictions_faceted(experiment_dir, show=False):
    """
    Creates a GRID of scatter plots (one per fold).
    """
    pred_path = os.path.join(experiment_dir, 'predictions.csv')
    if not os.path.exists(pred_path):
        print("No predictions.csv found.")
        return

    df = pd.read_csv(pred_path)

    # 1. Determine Range for the diagonal line (Unified across all folds)
    min_val = min(df['Actual'].min(), df['Predicted'].min())
    max_val = max(df['Actual'].max(), df['Predicted'].max())

    # 2. Create the FacetGrid
    # col_wrap=3 means it will put 3 plots per row, then start a new row
    g = sns.relplot(
        data=df,
        x='Actual', y='Predicted',
        col='Fold', col_wrap=3,
        kind='scatter',
        height=4, aspect=1,
        alpha=0.3, s=10,
        color='#2b7bba'  # Consistent color
    )

    # 3. Add the Perfect Fit Diagonal Line to EVERY subplot
    def plot_diagonal(x, y, **kwargs):
        # Draw line from min to max
        plt.plot([min_val, max_val], [min_val, max_val],
                 color='red', linestyle='--', linewidth=1.5)

    # Map the diagonal function to the grid
    g.map(plot_diagonal, 'Actual', 'Predicted')

    # 4. Cleanup titles and axis labels
    g.fig.suptitle("Predicted vs Actual (Per Fold Analysis)", y=1.02, fontsize=16)
    g.set_axis_labels("Actual Power (kW)", "Predicted Power (kW)")

    # Save
    save_path = os.path.join(experiment_dir, 'plot_scatter_faceted.png')
    plt.savefig(save_path, bbox_inches='tight')  # bbox_inches ensures title isn't cut off
    print(f"Saved Faceted Plot: {save_path}")

    if show:
        plt.show()
    plt.close()


dir = "results/Solar_Energy_Generation_CK2/ANN/20251207_000826"
show_plots = True

print(f"--- Analyzing: {dir} ---")
losses, folds, summary = load_experiment_data(dir)

plot_training_losses(losses, dir, show=show_plots)
plot_metrics_distributions(folds, dir, show=show_plots)
plot_summary_bar(summary, dir, show=show_plots)
plot_predictions_scatter(dir, show=show_plots)
plot_predictions_faceted(dir, show=show_plots)
print("Analysis Complete.\n")
