#!/usr/bin/env python3
"""Plot CNN training history: loss and accuracy curves over epochs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("outputs/models/cnn_genre_resnet_medium/cnn_genre_history.csv"),
        help="Training history CSV produced by train_cnn_genre.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/reports/figures/training_history.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    history = pd.read_csv(args.history)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CNN/ResNet Genre Classifier — Training History", fontsize=14, fontweight="bold")

    # ── Loss curve ──────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss", color="#2196F3")
    ax.plot(history["epoch"], history["val_loss"], marker="s", label="Val loss", color="#F44336", linestyle="--")

    best_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
    ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1.2, label=f"Best epoch ({best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(history["epoch"])

    # ── Accuracy curve ───────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(history["epoch"], history["train_accuracy"], marker="o", label="Train accuracy", color="#2196F3")
    ax.plot(
        history["epoch"], history["val_accuracy"], marker="s", label="Val accuracy", color="#F44336", linestyle="--"
    )

    best_val_acc = history["val_accuracy"].max()
    best_val_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    ax.axvline(
        best_val_epoch,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label=f"Best epoch ({best_val_epoch}): {best_val_acc:.3f}",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(history["epoch"])
    ax.set_ylim(0, 1)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved training history plot → {args.output}")
    print(f"  Epochs run:          {len(history)}")
    print(f"  Best val accuracy:   {best_val_acc:.4f}  (epoch {best_val_epoch})")
    print(f"  Final train loss:    {history['train_loss'].iloc[-1]:.4f}")
    print(f"  Final val loss:      {history['val_loss'].iloc[-1]:.4f}")


if __name__ == "__main__":
    main()
