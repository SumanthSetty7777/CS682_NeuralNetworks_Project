#!/usr/bin/env python3
"""Bootstrap confidence intervals for Precision@K across representations.

Loads the aligned embeddings, runs nearest-neighbour retrieval to get
per-query precision values, then bootstraps 1000 times to produce 95% CIs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


MOOD_PROXY_COLUMNS = [
    "rms_mean", "rms_std", "zcr_mean",
    "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
]

REAL_METHODS_ORDER = ["mfcc_only", "handcrafted_full", "cnn_resnet_medium", "vggish_pretrained"]
LABELS = {
    "mfcc_only":          "MFCC only",
    "handcrafted_full":   "Handcrafted full",
    "cnn_resnet_medium":  "CNN/ResNet",
    "vggish_pretrained":  "VGGish",
}
COLORS = {
    "mfcc_only":          "#4CAF50",
    "handcrafted_full":   "#2196F3",
    "cnn_resnet_medium":  "#FF9800",
    "vggish_pretrained":  "#9C27B0",
}
TARGET_LABELS = {"genre": "Genre P@K", "rhythm": "Rhythm P@K", "mood_proxy": "Mood P@K"}


def scaled(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[cols].to_numpy(dtype=np.float32))


def embedding_cols(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("emb_")]


def per_query_precision(
    features: np.ndarray,
    query_indices: np.ndarray,
    labels: np.ndarray,
    tempo: np.ndarray,
    mood_matrix: np.ndarray,
    mood_relevant_indices: np.ndarray,
    k: int,
    tempo_threshold: float,
) -> dict[str, np.ndarray]:
    model = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    model.fit(features)
    rows = model.kneighbors(features[query_indices], return_distance=False)

    genre_prec = []
    rhythm_prec = []
    mood_prec = []

    for qi, (query_idx, row) in enumerate(zip(query_indices, rows)):
        neighbors = [idx for idx in row if idx != query_idx][:k]
        genre_prec.append(np.mean(labels[neighbors] == labels[query_idx]))
        rhythm_prec.append(np.mean(np.abs(tempo[neighbors] - tempo[query_idx]) <= tempo_threshold))
        mood_relevant = set(mood_relevant_indices[qi])
        mood_prec.append(np.mean([n in mood_relevant for n in neighbors]))

    return {
        "genre": np.array(genre_prec),
        "rhythm": np.array(rhythm_prec),
        "mood_proxy": np.array(mood_prec),
    }


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = (1 - ci) / 2 * 100
    upper = (1 + ci) / 2 * 100
    return float(np.mean(values)), float(np.percentile(means, lower)), float(np.percentile(means, upper))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--tempo-threshold", type=float, default=10.0)
    parser.add_argument("--mood-percentile", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path,
                        default=Path("outputs/reports/tables/bootstrap_ci.csv"))
    parser.add_argument("--output-png", type=Path,
                        default=Path("outputs/reports/figures/bootstrap_ci.png"))
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

    split_df = pd.read_csv(args.split)
    test_ids = set(split_df[split_df["split"] == "test"]["track_id"].astype(int))
    query_indices = data.index[data["track_id"].astype(int).isin(test_ids)].to_numpy()
    print(f"Test queries: {len(query_indices)}")

    labels = data["genre_top"].to_numpy()
    tempo = data["tempo"].to_numpy(dtype=np.float32)

    # Mood reference neighbours (closest 5% in mood-proxy space)
    mood_matrix = scaled(data, MOOD_PROXY_COLUMNS)
    mood_k = max(args.k, int(round((len(data) - 1) * args.mood_percentile / 100)))
    print(f"Mood relevance: closest {args.mood_percentile}% = {mood_k} tracks/query")

    mood_model = NearestNeighbors(n_neighbors=min(mood_k + 1, len(data)), metric="euclidean")
    mood_model.fit(mood_matrix)
    mood_rows = mood_model.kneighbors(mood_matrix[query_indices], return_distance=False)
    mood_relevant = np.array([
        [idx for idx in row if idx != qi][:mood_k]
        for qi, row in zip(query_indices, mood_rows)
    ], dtype=object)

    feature_columns = [
        c for c in handcrafted.columns
        if c != "track_id" and not c.startswith("cnn_") and not c.startswith("vggish_")
    ]
    mfcc_columns = [c for c in feature_columns if c.startswith("mfcc_")]

    representations = {
        "mfcc_only": scaled(data, mfcc_columns),
        "handcrafted_full": scaled(data, feature_columns),
        "cnn_resnet_medium": scaled(data, [f"cnn_{c}" for c in cnn_cols]),
        "vggish_pretrained": scaled(data, [f"vggish_{c}" for c in vggish_cols]),
    }

    rows = []
    ci_data: dict[str, dict[str, tuple[float, float, float]]] = {}

    for name, matrix in representations.items():
        print(f"Computing bootstrap CI for {name}...")
        precisions = per_query_precision(
            features=matrix,
            query_indices=query_indices,
            labels=labels,
            tempo=tempo,
            mood_matrix=mood_matrix,
            mood_relevant_indices=mood_relevant,
            k=args.k,
            tempo_threshold=args.tempo_threshold,
        )
        ci_data[name] = {}
        for target, values in precisions.items():
            mean, lo, hi = bootstrap_ci(values, args.n_bootstrap, args.ci, seed=args.seed)
            ci_data[name][target] = (mean, lo, hi)
            rows.append({
                "representation": name,
                "target": target,
                "mean": mean,
                "ci_lower": lo,
                "ci_upper": hi,
                "ci_width": hi - lo,
                "n_queries": len(values),
                "n_bootstrap": args.n_bootstrap,
                "confidence": args.ci,
            })
            print(f"  {target:12s}  mean={mean:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved {args.output_csv}")

    # ── Markdown summary table ────────────────────────────────────────────────
    md_path = args.output_csv.parent / "bootstrap_ci.md"
    lines = [
        f"| Representation | Genre P@{args.k} (95% CI) | Rhythm P@{args.k} (95% CI) | Mood P@{args.k} (95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for name in REAL_METHODS_ORDER:
        if name not in ci_data:
            continue
        cells = []
        for target in ["genre", "rhythm", "mood_proxy"]:
            mean, lo, hi = ci_data[name][target]
            cells.append(f"{mean:.3f} [{lo:.3f}–{hi:.3f}]")
        lines.append(f"| {LABELS[name]} | {' | '.join(cells)} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {md_path}")

    # ── Plot: error bar chart ─────────────────────────────────────────────────
    targets = ["genre", "rhythm", "mood_proxy"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Precision@{args.k} with 95% Bootstrap Confidence Intervals\n"
        f"({args.n_bootstrap} bootstrap samples, {len(query_indices)} test queries)",
        fontsize=13, fontweight="bold",
    )

    for ax, target in zip(axes, targets):
        names = [n for n in REAL_METHODS_ORDER if n in ci_data]
        means = [ci_data[n][target][0] for n in names]
        lows  = [ci_data[n][target][0] - ci_data[n][target][1] for n in names]
        highs = [ci_data[n][target][2] - ci_data[n][target][0] for n in names]
        colors = [COLORS[n] for n in names]
        x = np.arange(len(names))

        ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
        ax.errorbar(
            x, means,
            yerr=[lows, highs],
            fmt="none", color="black", capsize=5, linewidth=1.5,
        )
        for xi, (mean, lo_err, hi_err) in enumerate(zip(means, lows, highs)):
            ax.text(xi, mean + hi_err + 0.005, f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[n] for n in names], fontsize=9, rotation=12)
        ax.set_ylabel(f"Precision@{args.k}")
        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(means) + max(highs) + 0.12)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output_png}")


if __name__ == "__main__":
    main()
