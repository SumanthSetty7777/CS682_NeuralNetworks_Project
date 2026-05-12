#!/usr/bin/env python3
"""Comparison plots that separate oracle baselines from real methods.

tempo_only and mood_proxy_reference are oracle references (they use exactly
the feature being tested) and are shown separately from the real competing
methods. Produces three figures:
  1. Real methods comparison (MFCC, Handcrafted, CNN, VGGish) vs random baseline
  2. Oracle ceiling reference chart (tempo_only, mood_proxy_reference)
  3. Grouped bar chart: each method side-by-side for all 3 similarity targets
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


REAL_METHODS = ["random_baseline", "mfcc_only", "handcrafted_full", "cnn_resnet_medium", "vggish_pretrained"]
ORACLE_REFS  = ["tempo_only", "mood_proxy_reference"]

LABELS = {
    "random_baseline":    "Random\nbaseline",
    "mfcc_only":          "MFCC only",
    "handcrafted_full":   "Handcrafted\nfull",
    "cnn_resnet_medium":  "CNN/ResNet",
    "vggish_pretrained":  "VGGish",
    "tempo_only":         "Tempo only\n(oracle)",
    "mood_proxy_reference": "Mood proxy\n(oracle)",
}

COLORS = {
    "random_baseline":    "#9E9E9E",
    "mfcc_only":          "#4CAF50",
    "handcrafted_full":   "#2196F3",
    "cnn_resnet_medium":  "#FF9800",
    "vggish_pretrained":  "#9C27B0",
    "tempo_only":         "#00BCD4",
    "mood_proxy_reference": "#F44336",
}

TARGET_LABELS = {"genre": "Genre P@K", "rhythm": "Rhythm P@K", "mood_proxy": "Mood P@K"}
TARGET_COLORS = {"genre": "#3F51B5", "rhythm": "#4CAF50", "mood_proxy": "#FF9800"}


def get_precision(metrics: pd.DataFrame, representation: str, target: str, k: int) -> float:
    row = metrics[
        (metrics["representation"] == representation) &
        (metrics["target"] == target) &
        (metrics["metric"] == "precision_at_k") &
        (metrics["k"] == k)
    ]
    return float(row["value"].iloc[0]) if len(row) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/figures"))
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    k = args.k
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = ["genre", "rhythm", "mood_proxy"]

    # ── Figure 1: Real methods only (3 subplots, one per target) ────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Retrieval Precision@{k} — Real Methods",
        fontsize=13, fontweight="bold",
    )

    for ax, target in zip(axes, targets):
        reps = REAL_METHODS
        values = [get_precision(metrics, r, target, k) for r in reps]
        colors = [COLORS[r] for r in reps]
        bars = ax.bar([LABELS[r] for r in reps], values, color=colors, edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Precision@{k}")
        ax.set_ylim(0, min(1.05, max(values) + 0.15))
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = args.output_dir / "clean_real_methods_precision.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 2: Oracle reference bar ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle(
        f"Oracle Reference Ceilings (Precision@{k})",
        fontsize=12, fontweight="bold",
    )

    for ax, target in zip(axes, targets):
        reps = ORACLE_REFS
        values = [get_precision(metrics, r, target, k) for r in reps]
        colors = [COLORS[r] for r in reps]
        bars = ax.bar([LABELS[r] for r in reps], values, color=colors, edgecolor="white", linewidth=0.8, hatch="//")

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Precision@{k}")
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out = args.output_dir / "clean_oracle_reference.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Figure 3: Grouped bar — all real methods × all targets ───────────────
    reps = REAL_METHODS
    n_reps = len(reps)
    n_targets = len(targets)
    x = np.arange(n_reps)
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, target in enumerate(targets):
        values = [get_precision(metrics, r, target, k) for r in reps]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, values, width,
            label=TARGET_LABELS[target],
            color=TARGET_COLORS[target],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in reps], fontsize=10)
    ax.set_ylabel(f"Precision@{k}", fontsize=11)
    ax.set_title(
        f"Precision@{k} Across All Similarity Dimensions",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.75)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = args.output_dir / "clean_grouped_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
