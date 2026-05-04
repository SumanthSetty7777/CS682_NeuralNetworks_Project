#!/usr/bin/env python3
"""Create report-ready plots for the final medium evaluation outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPRESENTATION_LABELS = {
    "random_baseline": "Random",
    "tempo_only": "Tempo only",
    "mood_proxy_reference": "Mood proxy ref.",
    "mfcc_only": "MFCC only",
    "handcrafted_full": "Handcrafted full",
    "cnn_resnet_medium": "CNN/ResNet",
    "vggish_pretrained": "VGGish",
}

REPRESENTATION_ORDER = [
    "random_baseline",
    "tempo_only",
    "mood_proxy_reference",
    "mfcc_only",
    "handcrafted_full",
    "cnn_resnet_medium",
    "vggish_pretrained",
]

MODEL_REPRESENTATION_ORDER = [
    "random_baseline",
    "mfcc_only",
    "handcrafted_full",
    "cnn_resnet_medium",
    "vggish_pretrained",
]

TARGET_LABELS = {
    "genre": "Genre",
    "rhythm": "Rhythm",
    "mood_proxy": "Mood Proxy",
    "genre_mismatch": "Genre Mismatch",
    "tempo_difference": "Tempo Difference",
    "mood_proxy_distance": "Mood-Proxy Distance",
}


def add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["representation_label"] = frame["representation"].map(REPRESENTATION_LABELS)
    frame["target_label"] = frame["target"].map(TARGET_LABELS)
    return frame


def save_barplot(
    frame: pd.DataFrame,
    output: Path,
    title: str,
    y_label: str,
    hue_order: list[str],
) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    sns.barplot(
        data=frame,
        x="target_label",
        y="value",
        hue="representation",
        hue_order=hue_order,
        ax=axis,
        palette="tab10",
    )
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel(y_label)
    axis.set_ylim(0, 1.05)

    handles, labels = axis.get_legend_handles_labels()
    display_labels = [REPRESENTATION_LABELS[label] for label in labels]
    axis.legend(handles, display_labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_retrieval(metrics: pd.DataFrame, output_dir: Path, k: int) -> None:
    filtered = metrics[
        (metrics["k"] == k)
        & (metrics["metric"].isin(["precision_at_k", "map_at_k"]))
    ].copy()
    filtered = add_labels(filtered)

    precision = filtered[filtered["metric"] == "precision_at_k"]
    save_barplot(
        precision,
        output_dir / f"final_precision_at_{k}.png",
        f"Nearest-Neighbor Retrieval Precision@{k}",
        f"Precision@{k}",
        REPRESENTATION_ORDER,
    )

    map_frame = filtered[filtered["metric"] == "map_at_k"]
    save_barplot(
        map_frame,
        output_dir / f"final_map_at_{k}.png",
        f"Nearest-Neighbor Retrieval mAP@{k}",
        f"mAP@{k}",
        REPRESENTATION_ORDER,
    )

    macro = metrics[
        (metrics["k"] == k)
        & (metrics["target"] == "genre")
        & (metrics["metric"] == "macro_precision_at_k")
    ].copy()
    macro = add_labels(macro)
    save_barplot(
        macro,
        output_dir / f"final_macro_genre_precision_at_{k}.png",
        f"Macro Genre Precision@{k}",
        f"Macro Precision@{k}",
        REPRESENTATION_ORDER,
    )


def plot_correlations(correlations: pd.DataFrame, output_dir: Path) -> None:
    frame = add_labels(correlations)
    sns.set_theme(style="whitegrid", context="paper")
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    sns.barplot(
        data=frame,
        x="target_label",
        y="spearman_r",
        hue="representation",
        hue_order=[name for name in REPRESENTATION_ORDER if name != "random_baseline"],
        ax=axis,
        palette="tab10",
    )
    axis.set_title("Embedding Distance vs. Target Distance")
    axis.set_xlabel("")
    axis.set_ylabel("Spearman rho")
    axis.set_ylim(0, 1.05)

    handles, labels = axis.get_legend_handles_labels()
    display_labels = [REPRESENTATION_LABELS[label] for label in labels]
    axis.legend(handles, display_labels, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    output = output_dir / "final_distance_correlations.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def plot_per_genre(per_genre: pd.DataFrame, output_dir: Path, k: int) -> None:
    frame = per_genre[
        (per_genre["k"] == k)
        & (per_genre["representation"].isin(MODEL_REPRESENTATION_ORDER))
    ].copy()
    frame["representation_label"] = frame["representation"].map(REPRESENTATION_LABELS)
    genre_order = (
        frame[frame["representation"] == "vggish_pretrained"]
        .sort_values("precision_at_k", ascending=False)["genre"]
        .tolist()
    )

    sns.set_theme(style="whitegrid", context="paper")
    fig, axis = plt.subplots(figsize=(11.5, 6.4))
    sns.barplot(
        data=frame,
        y="genre",
        x="precision_at_k",
        hue="representation",
        hue_order=MODEL_REPRESENTATION_ORDER,
        order=genre_order,
        ax=axis,
        palette="tab10",
    )
    axis.set_title(f"Per-Genre Genre Retrieval Precision@{k}")
    axis.set_xlabel(f"Precision@{k}")
    axis.set_ylabel("")
    axis.set_xlim(0, 1.0)

    handles, labels = axis.get_legend_handles_labels()
    display_labels = [REPRESENTATION_LABELS[label] for label in labels]
    axis.legend(handles, display_labels, loc="lower right", frameon=True)
    fig.tight_layout()
    output = output_dir / f"final_per_genre_precision_at_{k}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--per-genre", type=Path, required=True)
    parser.add_argument("--correlations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    per_genre = pd.read_csv(args.per_genre)
    correlations = pd.read_csv(args.correlations)

    plot_retrieval(metrics, args.output_dir, args.k)
    plot_correlations(correlations, args.output_dir)
    plot_per_genre(per_genre, args.output_dir, args.k)


if __name__ == "__main__":
    main()
