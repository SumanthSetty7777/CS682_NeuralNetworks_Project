#!/usr/bin/env python3
"""Generate qualitative nearest-neighbor examples for the final report."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
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
    "handcrafted_full": "Handcrafted full",
    "cnn_resnet_medium": "CNN/ResNet",
    "vggish_pretrained": "VGGish",
}


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("emb_")]


def scaled_matrix(data: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return StandardScaler().fit_transform(data[columns].to_numpy(dtype=np.float32))


def load_track_metadata(tracks_csv: Path | None) -> pd.DataFrame:
    if tracks_csv is None or not tracks_csv.exists():
        return pd.DataFrame(columns=["track_id", "artist_name", "track_title"])

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    metadata = pd.DataFrame({"track_id": tracks.index.astype(int)})
    if ("artist", "name") in tracks.columns:
        metadata["artist_name"] = tracks[("artist", "name")].fillna("").astype(str).to_numpy()
    else:
        metadata["artist_name"] = ""

    if ("track", "title") in tracks.columns:
        metadata["track_title"] = tracks[("track", "title")].fillna("").astype(str).to_numpy()
    else:
        metadata["track_title"] = ""

    return metadata


def clean_text(value: object) -> str:
    text = html.unescape(str(value))
    return text.replace("|", "/").replace("\n", " ").strip()


def load_aligned_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    manifest = pd.read_csv(args.manifest)
    handcrafted = pd.read_csv(args.handcrafted)
    cnn = pd.read_csv(args.cnn_embeddings)
    pretrained = pd.read_csv(args.pretrained_embeddings)
    metadata = load_track_metadata(args.tracks_csv)

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
    data = data.merge(metadata, on="track_id", how="left")
    data[["artist_name", "track_title"]] = data[["artist_name", "track_title"]].fillna("")
    data = data.dropna().sort_values("track_id").reset_index(drop=True)

    feature_columns = [
        column
        for column in handcrafted.columns
        if column != "track_id"
    ]
    cnn_columns = [f"cnn_{column}" for column in embedding_columns(cnn)]
    vggish_columns = [f"vggish_{column}" for column in embedding_columns(pretrained)]

    representations = {
        "handcrafted_full": scaled_matrix(data, feature_columns),
        "cnn_resnet_medium": scaled_matrix(data, cnn_columns),
        "vggish_pretrained": scaled_matrix(data, vggish_columns),
    }
    return data, representations


def selected_query_indices(
    data: pd.DataFrame,
    split_path: Path,
    split_name: str,
    num_queries: int,
    seed: int,
) -> np.ndarray:
    split = pd.read_csv(split_path)
    split = split[split["split"] == split_name].copy()
    available = data[["track_id", "genre_top", "tempo"]].merge(split[["track_id"]], on="track_id", how="inner")
    if available.empty:
        raise ValueError("No query tracks matched the aligned data.")

    rng = np.random.default_rng(seed)
    selected_ids = []
    genre_order = available["genre_top"].value_counts().index.tolist()
    for genre in genre_order:
        candidates = available[available["genre_top"] == genre]
        if candidates.empty:
            continue
        selected_ids.append(int(candidates.sample(n=1, random_state=int(rng.integers(0, 1_000_000)))["track_id"].iloc[0]))
        if len(selected_ids) >= num_queries:
            break

    query_indices = data.index[data["track_id"].isin(selected_ids)].to_numpy()
    return query_indices[:num_queries]


def neighbor_rows(
    data: pd.DataFrame,
    representation_name: str,
    features: np.ndarray,
    query_indices: np.ndarray,
    mood_matrix: np.ndarray,
    k: int,
) -> list[dict[str, object]]:
    model = NearestNeighbors(n_neighbors=min(k + 1, len(data)), metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features[query_indices], return_distance=True)

    rows: list[dict[str, object]] = []
    for query_idx, distance_row, index_row in zip(query_indices, distances, indices, strict=True):
        query = data.iloc[query_idx]
        neighbors = [
            (float(distance), int(index))
            for distance, index in zip(distance_row, index_row, strict=True)
            if int(index) != int(query_idx)
        ][:k]

        for rank, (embedding_distance, neighbor_idx) in enumerate(neighbors, start=1):
            neighbor = data.iloc[neighbor_idx]
            rows.append(
                {
                    "query_track_id": int(query["track_id"]),
                    "query_artist": query["artist_name"],
                    "query_title": query["track_title"],
                    "query_genre": query["genre_top"],
                    "query_tempo": round(float(query["tempo"]), 2),
                    "representation": representation_name,
                    "representation_label": REPRESENTATION_LABELS[representation_name],
                    "rank": rank,
                    "neighbor_track_id": int(neighbor["track_id"]),
                    "neighbor_artist": neighbor["artist_name"],
                    "neighbor_title": neighbor["track_title"],
                    "neighbor_genre": neighbor["genre_top"],
                    "neighbor_tempo": round(float(neighbor["tempo"]), 2),
                    "same_genre": bool(query["genre_top"] == neighbor["genre_top"]),
                    "tempo_difference": round(abs(float(query["tempo"]) - float(neighbor["tempo"])), 2),
                    "mood_proxy_distance": round(float(np.linalg.norm(mood_matrix[query_idx] - mood_matrix[neighbor_idx])), 4),
                    "embedding_distance": round(embedding_distance, 4),
                }
            )
    return rows


def write_markdown(rows: pd.DataFrame, output: Path) -> None:
    lines = ["# Nearest-Neighbor Retrieval Examples", ""]
    query_columns = ["query_track_id", "query_artist", "query_title", "query_genre", "query_tempo"]
    for query_values, query_frame in rows.groupby(query_columns, sort=False):
        query_track_id, artist, title, genre, tempo = query_values
        artist = clean_text(artist)
        title = clean_text(title)
        display = f"{artist} - {title}".strip(" -")
        if not display:
            display = f"track {query_track_id}"
        lines.append(f"## Query {query_track_id}: {display}")
        lines.append("")
        lines.append(f"Genre: {genre} | Tempo: {tempo}")
        lines.append("")

        for representation, rep_frame in query_frame.groupby("representation_label", sort=False):
            lines.append(f"### {representation}")
            lines.append("")
            lines.append("| Rank | Track | Genre | Tempo | Same Genre | Tempo Diff | Mood Distance |")
            lines.append("|---:|---|---|---:|---|---:|---:|")
            for item in rep_frame.itertuples(index=False):
                track_name = f"{clean_text(item.neighbor_artist)} - {clean_text(item.neighbor_title)}".strip(" -")
                if not track_name:
                    track_name = f"track {item.neighbor_track_id}"
                lines.append(
                    "| "
                    f"{item.rank} | "
                    f"{item.neighbor_track_id}: {track_name} | "
                    f"{item.neighbor_genre} | "
                    f"{item.neighbor_tempo} | "
                    f"{item.same_genre} | "
                    f"{item.tempo_difference} | "
                    f"{item.mood_proxy_distance} |"
                )
            lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--handcrafted", type=Path, required=True)
    parser.add_argument("--cnn-embeddings", type=Path, required=True)
    parser.add_argument("--pretrained-embeddings", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--tracks-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=6)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, representations = load_aligned_data(args)
    query_indices = selected_query_indices(data, args.split, args.split_name, args.num_queries, args.seed)
    mood_matrix = scaled_matrix(data, MOOD_PROXY_COLUMNS)

    rows = []
    for representation_name, features in representations.items():
        rows.extend(neighbor_rows(data, representation_name, features, query_indices, mood_matrix, args.k))

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote {args.output_csv}")
    write_markdown(output, args.output_md)
    print(f"Aligned tracks: {len(data)}")
    print(f"Query tracks: {len(query_indices)}")


if __name__ == "__main__":
    main()
