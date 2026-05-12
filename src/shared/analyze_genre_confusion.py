#!/usr/bin/env python3
"""Genre confusion analysis — what genres does each method retrieve for each query genre?

For each representation, builds a confusion-style matrix:
  rows    = query genre
  columns = retrieved genre
  values  = fraction of top-10 neighbours that belong to that genre

This shows *which* genres get confused with each other under each embedding.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MOOD_PROXY_COLUMNS = [
    "rms_mean", "rms_std", "zcr_mean",
    "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
]

REAL_METHODS = {
    "handcrafted_full": "Handcrafted Full",
    "cnn_resnet_medium": "CNN/ResNet",
    "vggish_pretrained": "VGGish",
}


def scaled(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[cols].to_numpy(dtype=np.float32))


def embedding_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("emb_")]


def build_confusion_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    query_indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    unique_genres = sorted(set(labels))
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    n_genres = len(unique_genres)

    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(features)
    neighbor_idx_matrix = model.kneighbors(features[query_indices], return_distance=False)

    confusion = np.zeros((n_genres, n_genres), dtype=np.float32)
    counts = np.zeros(n_genres, dtype=np.float32)

    for qi, query_idx in enumerate(query_indices):
        query_genre_idx = genre_to_idx[labels[query_idx]]
        neighbors = [idx for idx in neighbor_idx_matrix[qi] if idx != query_idx][:k]
        counts[query_genre_idx] += 1
        for neighbor_idx in neighbors:
            retrieved_genre_idx = genre_to_idx[labels[neighbor_idx]]
            confusion[query_genre_idx, retrieved_genre_idx] += 1

    # Normalise each row by (count * k) to get fractions
    row_totals = counts[:, None] * k
    row_totals[row_totals == 0] = 1
    confusion = confusion / row_totals

    return confusion, unique_genres


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="cnn_genre_split.csv — use test queries only")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/figures"))
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/reports/tables/genre_confusion.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data...")
    manifest = pd.read_csv(args.manifest)[["track_id", "genre_top"]]
    handcrafted = pd.read_csv(args.handcrafted)
    cnn_emb = pd.read_csv(args.cnn_embeddings)
    vggish_emb = pd.read_csv(args.pretrained_embeddings)

    cnn_cols = embedding_cols(cnn_emb)
    vggish_cols = embedding_cols(vggish_emb)

    data = manifest.merge(handcrafted, on="track_id", how="inner")
    data = data.merge(
        cnn_emb[["track_id"] + cnn_cols].rename(columns={c: f"cnn_{c}" for c in cnn_cols}),
        on="track_id", how="inner",
    )
    data = data.merge(
        vggish_emb[["track_id"] + vggish_cols].rename(columns={c: f"vggish_{c}" for c in vggish_cols}),
        on="track_id", how="inner",
    )
    data = data.dropna().sort_values("track_id").reset_index(drop=True)
    print(f"Aligned tracks: {len(data)}")

    # Load test query indices
    split_df = pd.read_csv(args.split)
    test_ids = set(split_df[split_df["split"] == "test"]["track_id"].astype(int))
    query_indices = data.index[data["track_id"].astype(int).isin(test_ids)].to_numpy()
    print(f"Test query tracks: {len(query_indices)}")

    labels = data["genre_top"].to_numpy()

    feature_columns = [
        c for c in handcrafted.columns
        if c != "track_id" and not c.startswith("cnn_") and not c.startswith("vggish_")
    ]

    representations = {
        "handcrafted_full": scaled(data, feature_columns),
        "cnn_resnet_medium": scaled(data, [f"cnn_{c}" for c in cnn_cols]),
        "vggish_pretrained": scaled(data, [f"vggish_{c}" for c in vggish_cols]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        f"Genre Retrieval Confusion Matrix (top-{args.k} neighbours)\n"
        "Each row = query genre. Each cell = fraction of retrieved songs from that genre.\n"
        "Diagonal = correct retrievals. Off-diagonal = confusions.",
        fontsize=12, fontweight="bold",
    )

    all_rows = []

    for ax, (name, matrix) in zip(axes, representations.items()):
        title = REAL_METHODS[name]
        print(f"Building confusion matrix for {name}...")

        confusion, genres = build_confusion_matrix(matrix, labels, query_indices, k=args.k)

        # Shorten long genre names for display
        short_genres = [g.replace("Old-Time / Historic", "Old-Time").replace("Easy Listening", "Easy List.") for g in genres]

        sns.heatmap(
            confusion,
            ax=ax,
            xticklabels=short_genres,
            yticklabels=short_genres,
            cmap="YlOrRd",
            vmin=0, vmax=0.6,
            annot=True, fmt=".2f",
            annot_kws={"size": 6},
            linewidths=0.3,
            cbar_kws={"label": "Fraction of retrieved songs"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Retrieved genre", fontsize=9)
        ax.set_ylabel("Query genre", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelrotation=0, labelsize=7)

        for i, query_genre in enumerate(genres):
            for j, ret_genre in enumerate(genres):
                all_rows.append({
                    "representation": name,
                    "query_genre": query_genre,
                    "retrieved_genre": ret_genre,
                    "fraction": float(confusion[i, j]),
                })

    plt.tight_layout()
    out = args.output_dir / "genre_confusion_matrix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)
    print(f"Saved {args.output_csv}")

    # ── Additional: top confusion pairs per method ───────────────────────────
    df = pd.DataFrame(all_rows)
    off_diag = df[df["query_genre"] != df["retrieved_genre"]].copy()
    print("\nTop confusion pairs per representation (off-diagonal, k=10):")
    for name in representations:
        subset = off_diag[off_diag["representation"] == name].nlargest(5, "fraction")
        print(f"\n  {REAL_METHODS[name]}:")
        for _, row in subset.iterrows():
            print(f"    {row['query_genre']:25s} → {row['retrieved_genre']:25s}  {row['fraction']:.3f}")


if __name__ == "__main__":
    main()
