#!/usr/bin/env python3
"""Plot retrieval metrics produced by evaluate_retrieval.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METRIC_TITLES = {
    "genre_precision": "Genre Precision@K",
    "rhythm_precision": "Rhythm Precision@K",
    "mood_proxy_precision": "Mood-Proxy Precision@K",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for axis, metric in zip(axes, METRIC_TITLES, strict=True):
        metric_df = metrics[metrics["metric"] == metric]
        sns.barplot(
            data=metric_df,
            x="k",
            y="value",
            hue="representation",
            ax=axis,
        )
        axis.set_title(METRIC_TITLES[metric])
        axis.set_xlabel("K")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.05)
        axis.legend_.remove()

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
