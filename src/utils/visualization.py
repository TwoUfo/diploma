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
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, [h["train_loss"] for h in history], label="Train")
    axes[0, 0].plot(epochs, [h["val_loss"] for h in history], label="Val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs, [h["train_class_accuracy"] for h in history], label="Train")
    axes[0, 1].plot(epochs, [h["val_class_accuracy"] for h in history], label="Val")
    axes[0, 1].set_title("Classification Accuracy")
    axes[0, 1].legend()
    axes[0, 1].set_xlabel("Epoch")

    axes[1, 0].plot(epochs, [h["train_h_r2"] for h in history], label="Train")
    axes[1, 0].plot(epochs, [h["val_h_r2"] for h in history], label="Val")
    axes[1, 0].set_title("H Magnitude R²")
    axes[1, 0].legend()
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, [h["train_diam_r2"] for h in history], label="Train")
    axes[1, 1].plot(epochs, [h["val_diam_r2"] for h in history], label="Val")
    axes[1, 1].set_title("Diameter R²")
    axes[1, 1].legend()
    axes[1, 1].set_xlabel("Epoch")

    plt.suptitle("Training History", fontsize=14)
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
