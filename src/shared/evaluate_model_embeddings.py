#!/usr/bin/env python3
"""Evaluate a learned/pretrained embedding CSV against music similarity targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
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


def nearest_neighbor_indices(
    features: np.ndarray,
    max_k: int,
    query_indices: np.ndarray | None = None,
) -> np.ndarray:
    n_neighbors = min(max_k + 1, len(features))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    model.fit(features)
    queries = features if query_indices is None else features[query_indices]
    indices = model.kneighbors(queries, return_distance=False)

    if query_indices is None:
        query_indices = np.arange(len(features))

    rows = []
    for query_idx, row in zip(query_indices, indices, strict=True):
        rows.append([idx for idx in row if idx != query_idx][:max_k])

    return np.asarray(rows, dtype=int)


def precision_at_k(
    relevant: np.ndarray,
    neighbor_indices: np.ndarray,
    query_indices: np.ndarray,
    k: int,
) -> float:
    hits = relevant[query_indices[:, None], neighbor_indices[:, :k]]
    return float(np.mean(np.mean(hits, axis=1)))


def scaled_matrix(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = data[columns].to_numpy(dtype=np.float32)
    return StandardScaler().fit_transform(values)


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
        raise ValueError("No query tracks matched the embedding table.")

    return query_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation-name", type=str, required=True)
    parser.add_argument("--ks", type=parse_ks, default=DEFAULT_KS)
    parser.add_argument("--tempo-threshold", type=float, default=10.0)
    parser.add_argument(
        "--mood-relevance-percentile",
        type=float,
        default=5.0,
        help="A track is mood-relevant if it is within this closest percent of mood-proxy neighbors.",
    )
    parser.add_argument(
        "--query-track-ids",
        type=Path,
        default=None,
        help="Optional CSV containing track_id rows to use as retrieval queries.",
    )
    parser.add_argument(
        "--query-split",
        type=str,
        default=None,
        help="Optional split value, such as test, when --query-track-ids has a split column.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = pd.read_csv(args.embeddings)
    target_features = pd.read_csv(args.target_features)
    manifest = pd.read_csv(args.manifest)

    data = embeddings.merge(manifest[["track_id", "genre_top"]], on="track_id", how="inner")
    data = data.merge(
        target_features[["track_id", "tempo", *MOOD_PROXY_COLUMNS]],
        on="track_id",
        how="inner",
    )
    data = data.dropna().sort_values("track_id").reset_index(drop=True)

    embedding_columns = [column for column in embeddings.columns if column != "track_id"]
    embedding_matrix = scaled_matrix(data, embedding_columns)
    labels = data["genre_top"].to_numpy()
    tempo = data["tempo"].to_numpy(dtype=np.float32)
    query_indices = load_query_indices(data, args.query_track_ids, args.query_split)

    max_k = max(args.ks)
    neighbor_indices = nearest_neighbor_indices(
        embedding_matrix,
        max_k=max_k,
        query_indices=query_indices,
    )

    same_genre = labels[:, None] == labels[None, :]
    rhythm_similar = np.abs(tempo[:, None] - tempo[None, :]) <= args.tempo_threshold

    mood_matrix = scaled_matrix(data, MOOD_PROXY_COLUMNS)
    mood_relevant_count = max(
        max_k,
        int(round((len(data) - 1) * args.mood_relevance_percentile / 100)),
    )
    mood_relevant_neighbors = nearest_neighbor_indices(mood_matrix, mood_relevant_count)
    mood_relevant = np.zeros((len(data), len(data)), dtype=bool)
    mood_relevant[np.arange(len(data))[:, None], mood_relevant_neighbors] = True

    rows = []
    for k in args.ks:
        rows.extend(
            [
                {
                    "representation": args.representation_name,
                    "metric": "genre_precision",
                    "k": k,
                    "value": precision_at_k(same_genre, neighbor_indices, query_indices, k),
                },
                {
                    "representation": args.representation_name,
                    "metric": "rhythm_precision",
                    "k": k,
                    "value": precision_at_k(rhythm_similar, neighbor_indices, query_indices, k),
                },
                {
                    "representation": args.representation_name,
                    "metric": "mood_proxy_precision",
                    "k": k,
                    "value": precision_at_k(mood_relevant, neighbor_indices, query_indices, k),
                },
            ]
        )

    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"Evaluated candidate tracks: {len(data)}")
    print(f"Evaluated queries: {len(query_indices)}")
    print(f"Mood-proxy relevance: closest {args.mood_relevance_percentile:g}% ({mood_relevant_count} tracks/query)")
    print(results.pivot_table(index=["representation", "k"], columns="metric", values="value").round(4))


if __name__ == "__main__":
    main()
