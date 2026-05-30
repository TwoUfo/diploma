import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_class_distribution(labels: pd.Series, title: str = "Asteroid Class Distribution"):
    counts = labels.value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(counts)))
    ax.set_xlabel("Count")
    ax.set_title(title)
    for i, (val, name) in enumerate(zip(counts.values, counts.index)):
        ax.text(val + max(counts) * 0.01, i, f"{val:,}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_missing_heatmap(df: pd.DataFrame, title: str = "Missing Data Pattern"):
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    cols_with_missing = missing_pct[missing_pct > 0].index.tolist()
    if not cols_with_missing:
        print("No missing data found.")
        return None

    fig, ax = plt.subplots(figsize=(12, max(4, len(cols_with_missing) * 0.4)))
    missing_pct[cols_with_missing].plot(kind="barh", ax=ax, color="salmon")
    ax.set_xlabel("Missing Fraction")
    ax.set_title(title)
    for i, val in enumerate(missing_pct[cols_with_missing].values):
        ax.text(val + 0.01, i, f"{val:.1%}", va="center", fontsize=9)
    plt.tight_layout()
    return fig


def plot_training_history(history: list[dict]):
    """Plot per-epoch metrics for all four MTL tasks plus total loss.

    Layout: 2 × 3 grid — total loss, class accuracy, then R² for diameter,
    albedo, rotation period, and one spare panel for class F1.
    """
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    def _plot(ax, key, title):
        ax.plot(epochs, [h[f"train_{key}"] for h in history], label="Train")
        ax.plot(epochs, [h[f"val_{key}"] for h in history], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    _plot(axes[0, 0], "loss",            "Total Loss")
    _plot(axes[0, 1], "class_accuracy",  "Classification Accuracy")
    _plot(axes[0, 2], "class_f1_macro",  "Class F1 (macro)")
    _plot(axes[1, 0], "diam_r2",         "Diameter R²")
    _plot(axes[1, 1], "albedo_r2",       "Albedo R²")
    _plot(axes[1, 2], "rot_r2",          "Rotation Period R²")

    plt.suptitle("Training History — 4-task MTL", fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized by row)")
    plt.tight_layout()
    return fig
