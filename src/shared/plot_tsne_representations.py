#!/usr/bin/env python3
"""t-SNE visualizations for handcrafted, CNN, and VGGish representations.

Each representation gets a figure with 3 subplots: coloured by genre,
by tempo, and by mood-proxy PC1.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


MOOD_PROXY_COLUMNS = [
    "rms_mean", "rms_std", "zcr_mean",
    "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
]

GENRE_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]


def scaled(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[cols].to_numpy(dtype=np.float32))


def embedding_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("emb_")]


def balanced_sample(data: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genres = data["genre_top"].unique()
    per_genre = max(1, sample_size // len(genres))
    parts = []
    for genre in genres:
        subset = data[data["genre_top"] == genre]
        n = min(len(subset), per_genre)
        parts.append(subset.sample(n=n, random_state=int(rng.integers(0, 10_000))))
    return pd.concat(parts).reset_index(drop=True)


def run_tsne(matrix: np.ndarray, seed: int, perplexity: float = 30.0) -> np.ndarray:
    # PCA to 50 dims first (standard practice — speeds up t-SNE dramatically)
    n_components_pca = min(50, matrix.shape[1], matrix.shape[0] - 1)
    reduced = PCA(n_components=n_components_pca, random_state=seed).fit_transform(matrix)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        max_iter=1000,
    )
    return tsne.fit_transform(reduced)


def plot_representation(
    name: str,
    coords: np.ndarray,
    genres: np.ndarray,
    tempo: np.ndarray,
    mood_pc1: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    unique_genres = sorted(set(genres))
    genre_to_idx = {g: i for i, g in enumerate(unique_genres)}
    genre_indices = np.array([genre_to_idx[g] for g in genres])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"t-SNE — {title}", fontsize=14, fontweight="bold")

    # ── By genre ──────────────────────────────────────────────────────────
    ax = axes[0]
    for i, genre in enumerate(unique_genres):
        mask = genre_indices == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=GENRE_PALETTE[i % len(GENRE_PALETTE)],
            label=genre, s=4, alpha=0.6, linewidths=0,
        )
    ax.set_title("Coloured by Genre")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(
        loc="upper right", fontsize=5, markerscale=2,
        framealpha=0.7, ncol=2,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # ── By tempo ──────────────────────────────────────────────────────────
    ax = axes[1]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=tempo, cmap="viridis", s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Tempo (BPM)")
    ax.set_title("Coloured by Tempo")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_xticks([])
    ax.set_yticks([])

    # ── By mood-proxy PC1 ─────────────────────────────────────────────────
    ax = axes[2]
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=mood_pc1, cmap="magma", s=4, alpha=0.6, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Mood-Proxy PC1")
    ax.set_title("Coloured by Mood-Proxy (PC1)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reports/figures"))
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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
    data = data.dropna().reset_index(drop=True)
    print(f"Aligned tracks: {len(data)}")

    sample = balanced_sample(data, args.sample_size, args.seed)
    print(f"Sampled {len(sample)} tracks for t-SNE (balanced by genre)")

    # ── Shared auxiliary arrays ───────────────────────────────────────────
    genres = sample["genre_top"].to_numpy()
    tempo = sample["tempo"].to_numpy(dtype=np.float32)

    mood_raw = scaled(sample, MOOD_PROXY_COLUMNS)
    mood_pc1 = PCA(n_components=1, random_state=args.seed).fit_transform(mood_raw)[:, 0]

    # ── Feature columns ───────────────────────────────────────────────────
    feature_columns = [
        c for c in handcrafted.columns
        if c != "track_id" and not c.startswith("cnn_") and not c.startswith("vggish_")
    ]

    representations = {
        "handcrafted_full": (scaled(sample, feature_columns), "Handcrafted Features", "tsne_handcrafted_full.png"),
        "cnn_resnet_medium": (scaled(sample, [f"cnn_{c}" for c in cnn_cols]), "CNN/ResNet Embeddings", "tsne_cnn_resnet_medium.png"),
        "vggish_pretrained": (scaled(sample, [f"vggish_{c}" for c in vggish_cols]), "VGGish Pretrained Embeddings", "tsne_vggish_pretrained.png"),
    }

    for name, (matrix, title, filename) in representations.items():
        print(f"\nRunning t-SNE for {name} ({matrix.shape[0]} × {matrix.shape[1]})...")
        coords = run_tsne(matrix, seed=args.seed)
        plot_representation(
            name=name,
            coords=coords,
            genres=genres,
            tempo=tempo,
            mood_pc1=mood_pc1,
            output_path=args.output_dir / filename,
            title=title,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
