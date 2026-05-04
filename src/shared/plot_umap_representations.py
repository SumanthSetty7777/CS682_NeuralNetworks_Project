#!/usr/bin/env python3
"""Create UMAP visualizations for final music similarity representations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
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
    "handcrafted_full": "Handcrafted Full",
    "cnn_resnet_medium": "CNN/ResNet",
    "vggish_pretrained": "VGGish",
}


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def scaled_matrix(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[columns].to_numpy(dtype=np.float32))


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(args.manifest)
    handcrafted = pd.read_csv(args.handcrafted)
    cnn = pd.read_csv(args.cnn_embeddings)
    pretrained = pd.read_csv(args.pretrained_embeddings)

    data = manifest[["track_id", "genre_top"]].merge(handcrafted, on="track_id", how="inner")
    data = data.merge(
        cnn.rename(columns={column: f"cnn_{column}" for column in embedding_columns(cnn)}),
        on="track_id",
        how="inner",
    )
    data = data.merge(
        pretrained.rename(columns={column: f"vggish_{column}" for column in embedding_columns(pretrained)}),
        on="track_id",
        how="inner",
    )
    data = data.dropna().sort_values("track_id").reset_index(drop=True)

    feature_columns = [column for column in handcrafted.columns if column != "track_id"]
    cnn_columns = [f"cnn_{column}" for column in embedding_columns(cnn)]
    vggish_columns = [f"vggish_{column}" for column in embedding_columns(pretrained)]
    representations = {
        "handcrafted_full": scaled_matrix(data, feature_columns),
        "cnn_resnet_medium": scaled_matrix(data, cnn_columns),
        "vggish_pretrained": scaled_matrix(data, vggish_columns),
    }
    return data, representations


def add_mood_score(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    mood_matrix = scaled_matrix(data, MOOD_PROXY_COLUMNS)
    data["mood_proxy_pc1"] = PCA(n_components=1, random_state=0).fit_transform(mood_matrix).ravel()
    return data


def sample_indices(data: pd.DataFrame, sample_size: int, seed: int) -> np.ndarray:
    if sample_size >= len(data):
        return np.arange(len(data))
    sample = (
        data.groupby("genre_top", group_keys=False)
        .sample(frac=sample_size / len(data), random_state=seed)
        .index.to_numpy()
    )
    if len(sample) > sample_size:
        rng = np.random.default_rng(seed)
        sample = rng.choice(sample, size=sample_size, replace=False)
    return np.sort(sample)


def plot_representation(
    name: str,
    matrix: np.ndarray,
    sample_data: pd.DataFrame,
    sample_idx: np.ndarray,
    output_dir: Path,
    seed: int,
) -> None:
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    coordinates = reducer.fit_transform(matrix[sample_idx])
    frame = sample_data.copy()
    frame["umap_1"] = coordinates[:, 0]
    frame["umap_2"] = coordinates[:, 1]

    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    sns.scatterplot(
        data=frame,
        x="umap_1",
        y="umap_2",
        hue="genre_top",
        s=8,
        linewidth=0,
        alpha=0.75,
        ax=axes[0],
        legend=False,
        palette="tab20",
    )
    axes[0].set_title("Colored by Genre")

    tempo_plot = axes[1].scatter(
        frame["umap_1"],
        frame["umap_2"],
        c=frame["tempo"],
        s=8,
        alpha=0.75,
        cmap="viridis",
        linewidths=0,
    )
    axes[1].set_title("Colored by Tempo")
    fig.colorbar(tempo_plot, ax=axes[1], fraction=0.046, pad=0.04)

    mood_plot = axes[2].scatter(
        frame["umap_1"],
        frame["umap_2"],
        c=frame["mood_proxy_pc1"],
        s=8,
        alpha=0.75,
        cmap="magma",
        linewidths=0,
    )
    axes[2].set_title("Colored by Mood Proxy PC1")
    fig.colorbar(mood_plot, ax=axes[2], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(REPRESENTATION_LABELS[name], y=1.02)
    fig.tight_layout()
    output = output_dir / f"umap_{name}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, representations = load_data(args)
    data = add_mood_score(data)
    sample_idx = sample_indices(data, args.sample_size, args.seed)
    sample_data = data.iloc[sample_idx].reset_index(drop=True)

    print(f"Aligned tracks: {len(data)}")
    print(f"UMAP sample tracks: {len(sample_idx)}")
    for offset, (name, matrix) in enumerate(representations.items()):
        print(f"Plotting {name}...")
        plot_representation(name, matrix, sample_data, sample_idx, args.output_dir, args.seed + offset)


if __name__ == "__main__":
    main()
