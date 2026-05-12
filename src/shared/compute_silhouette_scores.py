#!/usr/bin/env python3
"""Compute silhouette scores for each audio representation, grouped by genre.

Silhouette score measures how well each track fits into its own genre cluster
compared to the nearest other genre cluster. Ranges from -1 (wrong cluster)
to +1 (tight, well-separated cluster). Higher = the embedding better organises
songs by genre in its space.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler


MOOD_PROXY_COLUMNS = [
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
]

REPRESENTATION_LABELS = {
    "mfcc_only": "MFCC only",
    "handcrafted_full": "Handcrafted full",
    "cnn_resnet_medium": "CNN/ResNet",
    "vggish_pretrained": "VGGish",
}


def scaled(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[columns].to_numpy(dtype=np.float32))


def embedding_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("emb_")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Max tracks to sample for silhouette computation (full set is slow).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/reports/tables/silhouette_scores.csv"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/reports/figures/silhouette_scores.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load and align all data ──────────────────────────────────────────────
    manifest = pd.read_csv(args.manifest)[["track_id", "genre_top"]]
    handcrafted = pd.read_csv(args.handcrafted)
    cnn_emb = pd.read_csv(args.cnn_embeddings)
    vggish_emb = pd.read_csv(args.pretrained_embeddings)

    data = manifest.merge(handcrafted, on="track_id", how="inner")
    cnn_prefixed = cnn_emb.rename(columns={c: f"cnn_{c}" for c in embedding_cols(cnn_emb)})
    vgg_prefixed = vggish_emb.rename(columns={c: f"vggish_{c}" for c in embedding_cols(vggish_emb)})
    data = data.merge(cnn_prefixed, on="track_id", how="inner")
    data = data.merge(vgg_prefixed, on="track_id", how="inner")
    data = data.dropna().sort_values("track_id").reset_index(drop=True)

    print(f"Aligned tracks: {len(data)}")

    # ── Sample for speed ─────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    if len(data) > args.sample_size:
        sample_idx = rng.choice(len(data), size=args.sample_size, replace=False)
        sample = data.iloc[sample_idx].reset_index(drop=True)
        print(f"Sampled {args.sample_size} tracks for silhouette computation")
    else:
        sample = data.copy()

    labels = sample["genre_top"].to_numpy()
    n_genres = len(np.unique(labels))
    print(f"Genres: {n_genres}  |  Sample size: {len(sample)}")

    # ── Build representations ─────────────────────────────────────────────────
    feature_columns = [
        c for c in data.columns
        if c not in {"track_id", "genre_top", "genre_id", "genre_title", "subset", "audio_path", "exists"}
        and not c.startswith("cnn_")
        and not c.startswith("vggish_")
    ]
    mfcc_columns = [c for c in feature_columns if c.startswith("mfcc_")]
    cnn_columns = [c for c in data.columns if c.startswith("cnn_emb_")]
    vggish_columns = [c for c in data.columns if c.startswith("vggish_emb_")]

    representations: dict[str, np.ndarray] = {
        "mfcc_only": scaled(sample, mfcc_columns),
        "handcrafted_full": scaled(sample, feature_columns),
        "cnn_resnet_medium": scaled(sample, cnn_columns),
        "vggish_pretrained": scaled(sample, vggish_columns),
    }

    # ── Compute silhouette scores ─────────────────────────────────────────────
    rows = []
    per_genre_rows = []

    for name, matrix in representations.items():
        print(f"Computing silhouette for {name}...")
        overall = float(silhouette_score(matrix, labels, metric="euclidean", sample_size=None))
        per_sample = silhouette_samples(matrix, labels, metric="euclidean")

        for genre in np.unique(labels):
            mask = labels == genre
            per_genre_rows.append(
                {
                    "representation": name,
                    "genre": genre,
                    "silhouette_score": float(np.mean(per_sample[mask])),
                    "count": int(mask.sum()),
                }
            )

        rows.append({"representation": name, "silhouette_score": overall})
        print(f"  Overall silhouette: {overall:.4f}")

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_csv, index=False)

    per_genre_path = args.output_csv.parent / "silhouette_scores_per_genre.csv"
    pd.DataFrame(per_genre_rows).to_csv(per_genre_path, index=False)
    print(f"Saved {args.output_csv}")
    print(f"Saved {per_genre_path}")

    # ── Plot 1: overall silhouette bar chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Silhouette Score by Genre Clustering", fontsize=14, fontweight="bold")

    labels_display = [REPRESENTATION_LABELS.get(r, r) for r in summary_df["representation"]]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    ax = axes[0]
    bars = ax.bar(labels_display, summary_df["silhouette_score"], color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    for bar, val in zip(bars, summary_df["silhouette_score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylabel("Silhouette Score (higher = better genre clustering)")
    ax.set_title("Overall Silhouette Score by Representation")
    ax.set_ylim(min(summary_df["silhouette_score"].min() - 0.05, -0.05), summary_df["silhouette_score"].max() + 0.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=15)

    # ── Plot 2: per-genre heatmap ─────────────────────────────────────────────
    per_genre_df = pd.DataFrame(per_genre_rows)
    pivot = per_genre_df.pivot(index="genre", columns="representation", values="silhouette_score")
    pivot = pivot[[c for c in ["mfcc_only", "handcrafted_full", "cnn_resnet_medium", "vggish_pretrained"] if c in pivot.columns]]
    pivot.columns = [REPRESENTATION_LABELS.get(c, c) for c in pivot.columns]

    ax = axes[1]
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.4)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Per-Genre Silhouette Score")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(val) < 0.25 else "white")

    plt.colorbar(im, ax=ax, label="Silhouette score")
    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved silhouette plot → {args.output_png}")

    # ── Also write markdown summary table ────────────────────────────────────
    md_path = args.output_csv.parent / "silhouette_scores.md"
    lines = ["| Representation | Silhouette Score |", "| --- | --- |"]
    for _, row in summary_df.iterrows():
        label = REPRESENTATION_LABELS.get(row["representation"], row["representation"])
        lines.append(f"| {label} | {row['silhouette_score']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
