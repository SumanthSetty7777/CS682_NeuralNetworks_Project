#!/usr/bin/env python3
"""Evaluate all report-facing music similarity representations together."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_KS = [5, 10, 20]
MOOD_PROXY_COLUMNS = [
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
]


def parse_ks(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def scaled_matrix(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = data[columns].to_numpy(dtype=np.float32)
    return StandardScaler().fit_transform(values)


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def load_query_indices(
    data: pd.DataFrame,
    query_track_ids: Path | None,
    query_split: str | None,
) -> np.ndarray:
    if query_track_ids is None:
        return np.arange(len(data))

    query_frame = pd.read_csv(query_track_ids)
    if query_split is not None:
        if "split" not in query_frame.columns:
            raise ValueError("--query-split requires a split column in --query-track-ids")
        query_frame = query_frame[query_frame["split"] == query_split]

    selected_ids = set(query_frame["track_id"].astype(int))
    query_indices = data.index[data["track_id"].astype(int).isin(selected_ids)].to_numpy()
    if len(query_indices) == 0:
        raise ValueError("No query tracks matched the aligned data.")

    return query_indices


def nearest_neighbor_indices(
    features: np.ndarray,
    max_k: int,
    query_indices: np.ndarray,
) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(max_k + 1, len(features)), metric="euclidean")
    model.fit(features)
    rows = model.kneighbors(features[query_indices], return_distance=False)

    neighbors = []
    for query_idx, row in zip(query_indices, rows, strict=True):
        neighbors.append([idx for idx in row if idx != query_idx][:max_k])
    return np.asarray(neighbors, dtype=int)


def random_neighbor_indices(
    n_items: int,
    max_k: int,
    query_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_items)
    rows = []
    for query_idx in query_indices:
        candidates = all_indices[all_indices != query_idx]
        rows.append(rng.choice(candidates, size=max_k, replace=False))
    return np.asarray(rows, dtype=int)


def rhythm_relevant_counts(tempo: np.ndarray, query_indices: np.ndarray, threshold: float) -> np.ndarray:
    sorted_tempo = np.sort(tempo)
    query_tempo = tempo[query_indices]
    lower = np.searchsorted(sorted_tempo, query_tempo - threshold, side="left")
    upper = np.searchsorted(sorted_tempo, query_tempo + threshold, side="right")
    return np.maximum(upper - lower - 1, 0)


def average_precision_at_k(
    hits: np.ndarray,
    relevant_counts: np.ndarray,
    k: int,
) -> np.ndarray:
    hits_k = hits[:, :k].astype(np.float32)
    cumulative_hits = np.cumsum(hits_k, axis=1)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    precision_at_rank = cumulative_hits / ranks
    denominators = np.minimum(relevant_counts.astype(np.float32), float(k))
    ap = np.full(len(hits_k), np.nan, dtype=np.float32)
    valid = denominators > 0
    ap[valid] = np.sum(precision_at_rank[valid] * hits_k[valid], axis=1) / denominators[valid]
    return ap


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    return np.mean(hits[:, :k], axis=1)


def macro_average_by_genre(values: np.ndarray, query_labels: np.ndarray) -> float:
    per_genre = []
    for genre in np.unique(query_labels):
        per_genre.append(float(np.mean(values[query_labels == genre])))
    return float(np.mean(per_genre))


def mood_relevance_neighbors(
    mood_matrix: np.ndarray,
    query_indices: np.ndarray,
    relevant_count: int,
) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=min(relevant_count + 1, len(mood_matrix)), metric="euclidean")
    model.fit(mood_matrix)
    rows = model.kneighbors(mood_matrix[query_indices], return_distance=False)

    neighbors = []
    for query_idx, row in zip(query_indices, rows, strict=True):
        neighbors.append([idx for idx in row if idx != query_idx][:relevant_count])
    return np.asarray(neighbors, dtype=int)


def mood_hits(neighbor_indices: np.ndarray, relevant_neighbors: np.ndarray) -> np.ndarray:
    rows = [
        np.isin(neighbors, relevant, assume_unique=False)
        for neighbors, relevant in zip(neighbor_indices, relevant_neighbors, strict=True)
    ]
    return np.asarray(rows, dtype=bool)


def build_representations(
    data: pd.DataFrame,
    cnn_embeddings: pd.DataFrame,
    pretrained_embeddings: pd.DataFrame,
) -> dict[str, np.ndarray | None]:
    feature_columns = [
        column
        for column in data.columns
        if column not in {"track_id", "genre_top", "genre_id", "genre_title", "subset", "audio_path", "exists"}
        and not column.startswith("cnn_")
        and not column.startswith("vggish_")
    ]
    mfcc_columns = [column for column in feature_columns if column.startswith("mfcc_")]
    cnn_columns = embedding_columns(cnn_embeddings)
    pretrained_columns = embedding_columns(pretrained_embeddings)

    return {
        "random_baseline": None,
        "tempo_only": scaled_matrix(data, ["tempo"]),
        "mood_proxy_reference": scaled_matrix(data, MOOD_PROXY_COLUMNS),
        "mfcc_only": scaled_matrix(data, mfcc_columns),
        "handcrafted_full": scaled_matrix(data, feature_columns),
        "cnn_resnet_medium": scaled_matrix(data, [f"cnn_{column}" for column in cnn_columns]),
        "vggish_pretrained": scaled_matrix(data, [f"vggish_{column}" for column in pretrained_columns]),
    }


def evaluate_neighbors(
    representation: str,
    neighbor_indices: np.ndarray,
    labels: np.ndarray,
    tempo: np.ndarray,
    query_indices: np.ndarray,
    query_labels: np.ndarray,
    mood_relevant_neighbors: np.ndarray,
    mood_relevant_count: int,
    ks: list[int],
    tempo_threshold: float,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    genre_hits = labels[neighbor_indices] == labels[query_indices, None]
    rhythm_hits = np.abs(tempo[neighbor_indices] - tempo[query_indices, None]) <= tempo_threshold
    mood_proxy_hits = mood_hits(neighbor_indices, mood_relevant_neighbors)

    genre_counts = pd.Series(labels).value_counts().reindex(labels[query_indices]).to_numpy() - 1
    rhythm_counts = rhythm_relevant_counts(tempo, query_indices, tempo_threshold)
    mood_counts = np.full(len(query_indices), mood_relevant_count, dtype=int)

    target_hits = {
        "genre": (genre_hits, genre_counts),
        "rhythm": (rhythm_hits, rhythm_counts),
        "mood_proxy": (mood_proxy_hits, mood_counts),
    }

    metric_rows: list[dict[str, float | int | str]] = []
    per_genre_rows: list[dict[str, float | int | str]] = []

    for target, (hits, relevant_counts) in target_hits.items():
        for k in ks:
            per_query_precision = precision_at_k(hits, k)
            per_query_ap = average_precision_at_k(hits, relevant_counts, k)

            metric_rows.append(
                {
                    "representation": representation,
                    "target": target,
                    "metric": "precision_at_k",
                    "k": k,
                    "value": float(np.mean(per_query_precision)),
                }
            )
            metric_rows.append(
                {
                    "representation": representation,
                    "target": target,
                    "metric": "map_at_k",
                    "k": k,
                    "value": float(np.nanmean(per_query_ap)),
                }
            )

            if target == "genre":
                metric_rows.append(
                    {
                        "representation": representation,
                        "target": target,
                        "metric": "macro_precision_at_k",
                        "k": k,
                        "value": macro_average_by_genre(per_query_precision, query_labels),
                    }
                )
                for genre in np.unique(query_labels):
                    mask = query_labels == genre
                    per_genre_rows.append(
                        {
                            "representation": representation,
                            "genre": genre,
                            "k": k,
                            "query_count": int(mask.sum()),
                            "precision_at_k": float(np.mean(per_query_precision[mask])),
                            "map_at_k": float(np.nanmean(per_query_ap[mask])),
                        }
                    )

    return metric_rows, per_genre_rows


def sampled_pair_correlations(
    representation: str,
    features: np.ndarray,
    labels: np.ndarray,
    tempo: np.ndarray,
    mood_matrix: np.ndarray,
    pair_sample_size: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    rng = np.random.default_rng(seed)
    n_items = len(features)
    left = rng.integers(0, n_items, size=pair_sample_size)
    right = rng.integers(0, n_items, size=pair_sample_size)
    same = left == right
    while same.any():
        right[same] = rng.integers(0, n_items, size=int(same.sum()))
        same = left == right

    representation_distance = np.linalg.norm(features[left] - features[right], axis=1)
    targets = {
        "genre_mismatch": (labels[left] != labels[right]).astype(np.float32),
        "tempo_difference": np.abs(tempo[left] - tempo[right]),
        "mood_proxy_distance": np.linalg.norm(mood_matrix[left] - mood_matrix[right], axis=1),
    }

    rows: list[dict[str, float | int | str]] = []
    for target, values in targets.items():
        statistic = spearmanr(representation_distance, values, nan_policy="omit")
        rows.append(
            {
                "representation": representation,
                "target": target,
                "spearman_r": float(statistic.statistic),
                "p_value": float(statistic.pvalue),
                "pair_sample_size": int(pair_sample_size),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--per-genre-output", type=Path, required=True)
    parser.add_argument("--correlations-output", type=Path, required=True)
    parser.add_argument("--ks", type=parse_ks, default=DEFAULT_KS)
    parser.add_argument("--tempo-threshold", type=float, default=10.0)
    parser.add_argument("--mood-relevance-percentile", type=float, default=5.0)
    parser.add_argument("--query-track-ids", type=Path, default=None)
    parser.add_argument("--query-split", type=str, default=None)
    parser.add_argument("--pair-sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    handcrafted = pd.read_csv(args.handcrafted)
    cnn_embeddings = pd.read_csv(args.cnn_embeddings)
    pretrained_embeddings = pd.read_csv(args.pretrained_embeddings)

    data = manifest[["track_id", "genre_top"]].merge(handcrafted, on="track_id", how="inner")
    cnn_prefixed = cnn_embeddings.rename(
        columns={column: f"cnn_{column}" for column in embedding_columns(cnn_embeddings)}
    )
    vggish_prefixed = pretrained_embeddings.rename(
        columns={column: f"vggish_{column}" for column in embedding_columns(pretrained_embeddings)}
    )
    data = data.merge(cnn_prefixed, on="track_id", how="inner")
    data = data.merge(vggish_prefixed, on="track_id", how="inner")
    data = data.dropna().sort_values("track_id").reset_index(drop=True)

    labels = data["genre_top"].to_numpy()
    tempo = data["tempo"].to_numpy(dtype=np.float32)
    query_indices = load_query_indices(data, args.query_track_ids, args.query_split)
    query_labels = labels[query_indices]
    max_k = max(args.ks)

    mood_matrix = scaled_matrix(data, MOOD_PROXY_COLUMNS)
    mood_relevant_count = max(
        max_k,
        int(round((len(data) - 1) * args.mood_relevance_percentile / 100)),
    )
    mood_neighbors = mood_relevance_neighbors(mood_matrix, query_indices, mood_relevant_count)

    representations = build_representations(data, cnn_embeddings, pretrained_embeddings)

    metric_rows: list[dict[str, float | int | str]] = []
    per_genre_rows: list[dict[str, float | int | str]] = []
    correlation_rows: list[dict[str, float | int | str]] = []

    for idx, (name, matrix) in enumerate(representations.items()):
        print(f"Evaluating {name}...")
        if matrix is None:
            neighbor_indices = random_neighbor_indices(len(data), max_k, query_indices, args.seed)
        else:
            neighbor_indices = nearest_neighbor_indices(matrix, max_k, query_indices)
            correlation_rows.extend(
                sampled_pair_correlations(
                    representation=name,
                    features=matrix,
                    labels=labels,
                    tempo=tempo,
                    mood_matrix=mood_matrix,
                    pair_sample_size=args.pair_sample_size,
                    seed=args.seed + idx,
                )
            )

        rows, genre_rows = evaluate_neighbors(
            representation=name,
            neighbor_indices=neighbor_indices,
            labels=labels,
            tempo=tempo,
            query_indices=query_indices,
            query_labels=query_labels,
            mood_relevant_neighbors=mood_neighbors,
            mood_relevant_count=mood_relevant_count,
            ks=args.ks,
            tempo_threshold=args.tempo_threshold,
        )
        metric_rows.extend(rows)
        per_genre_rows.extend(genre_rows)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.metrics_output, index=False)
    pd.DataFrame(per_genre_rows).to_csv(args.per_genre_output, index=False)
    pd.DataFrame(correlation_rows).to_csv(args.correlations_output, index=False)

    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {args.per_genre_output}")
    print(f"Wrote {args.correlations_output}")
    print(f"Aligned candidate tracks: {len(data)}")
    print(f"Evaluated queries: {len(query_indices)}")
    print(f"Mood-proxy relevance: closest {args.mood_relevance_percentile:g}% ({mood_relevant_count} tracks/query)")

    metrics = pd.DataFrame(metric_rows)
    print(
        metrics.pivot_table(
            index=["representation", "target", "k"],
            columns="metric",
            values="value",
        ).round(4)
    )


if __name__ == "__main__":
    main()
