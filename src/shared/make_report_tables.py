#!/usr/bin/env python3
"""Create compact CSV and Markdown tables for the final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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

CORRELATION_TARGETS = {
    "genre_mismatch": "Genre mismatch",
    "tempo_difference": "Tempo difference",
    "mood_proxy_distance": "Mood-proxy distance",
}


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for item in frame.itertuples(index=False):
        rows.append("| " + " | ".join(str(value) for value in item) + " |")
    return "\n".join(rows)


def write_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def k_metrics_table(metrics: pd.DataFrame, k: int) -> pd.DataFrame:
    filtered = metrics[metrics["k"] == k]

    def value(representation: str, target: str, metric: str) -> float:
        row = filtered[
            (filtered["representation"] == representation)
            & (filtered["target"] == target)
            & (filtered["metric"] == metric)
        ]
        if row.empty:
            return float("nan")
        return float(row["value"].iloc[0])

    rows = []
    for representation in REPRESENTATION_ORDER:
        rows.append(
            {
                "Representation": REPRESENTATION_LABELS[representation],
                f"Genre P@{k}": value(representation, "genre", "precision_at_k"),
                f"Genre mAP@{k}": value(representation, "genre", "map_at_k"),
                f"Macro Genre P@{k}": value(representation, "genre", "macro_precision_at_k"),
                f"Rhythm P@{k}": value(representation, "rhythm", "precision_at_k"),
                f"Rhythm mAP@{k}": value(representation, "rhythm", "map_at_k"),
                f"Mood P@{k}": value(representation, "mood_proxy", "precision_at_k"),
                f"Mood mAP@{k}": value(representation, "mood_proxy", "map_at_k"),
            }
        )

    return pd.DataFrame(rows).round(4)


def correlation_table(correlations: pd.DataFrame) -> pd.DataFrame:
    table = correlations.pivot_table(
        index="representation",
        columns="target",
        values="spearman_r",
    ).reindex(REPRESENTATION_ORDER)
    table = table.rename(index=REPRESENTATION_LABELS, columns=CORRELATION_TARGETS)
    table = table.reset_index().rename(columns={"representation": "Representation"})
    return table.round(4)


def training_table(metrics_json: Path) -> pd.DataFrame:
    metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
    rows = [
        {
            "Model": "CNN/ResNet medium",
            "Model size": metrics["model_size"],
            "Embedding dim": metrics["embedding_dim"],
            "Train tracks": metrics["train_tracks"],
            "Validation tracks": metrics["val_tracks"],
            "Test tracks": metrics["test_tracks"],
            "Epochs run": metrics["epochs_run"],
            "Best validation accuracy": metrics["best_val_accuracy"],
            "Test accuracy": metrics["test_accuracy"],
        }
    ]
    return pd.DataFrame(rows).round(4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--correlations", type=Path, required=True)
    parser.add_argument("--cnn-metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics)
    correlations = pd.read_csv(args.correlations)

    write_table(k_metrics_table(metrics, args.k), args.output_dir, f"final_k{args.k}_retrieval_metrics")
    write_table(correlation_table(correlations), args.output_dir, "final_distance_correlations")
    write_table(training_table(args.cnn_metrics_json), args.output_dir, "cnn_training_summary")


if __name__ == "__main__":
    main()
