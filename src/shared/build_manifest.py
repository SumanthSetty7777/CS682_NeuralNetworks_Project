#!/usr/bin/env python3
"""Build a clean manifest for the FMA small dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fma_audio_path(audio_root: Path, track_id: int) -> Path:
    """Return the canonical FMA audio path for a track id."""
    track = f"{track_id:06d}"
    return audio_root / track[:3] / f"{track}.mp3"


def load_tracks(tracks_csv: Path) -> pd.DataFrame:
    """Load FMA tracks.csv, whose columns use a two-row header."""
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    tracks.index.name = "track_id"
    return tracks


def load_genres(genres_csv: Path) -> pd.DataFrame:
    genres = pd.read_csv(genres_csv)
    genres = genres.rename(columns={"genre_id": "genre_id", "title": "genre_title"})
    return genres[["genre_id", "genre_title", "top_level"]]


def subset_values(subset: str) -> list[str]:
    if subset == "small":
        return ["small"]
    if subset == "medium":
        return ["small", "medium"]
    if subset == "large":
        return ["small", "medium", "large"]
    raise ValueError(f"Unknown subset: {subset}")


def build_manifest(tracks_csv: Path, genres_csv: Path, audio_root: Path, subset: str) -> pd.DataFrame:
    tracks = load_tracks(tracks_csv)
    selected = tracks[tracks[("set", "subset")].isin(subset_values(subset))].copy()

    manifest = pd.DataFrame(index=selected.index)
    manifest["track_id"] = selected.index.astype(int)
    manifest["subset"] = selected[("set", "subset")]
    manifest["genre_top"] = selected[("track", "genre_top")]

    genres = load_genres(genres_csv)
    top_level_genres = genres[genres["genre_id"] == genres["top_level"]].copy()
    manifest = manifest.merge(
        top_level_genres[["genre_id", "genre_title"]],
        left_on="genre_top",
        right_on="genre_title",
        how="left",
    )

    manifest["audio_path"] = manifest["track_id"].apply(
        lambda track_id: str(fma_audio_path(audio_root, int(track_id)))
    )
    manifest["exists"] = manifest["audio_path"].apply(lambda path: Path(path).exists())

    cols = ["track_id", "subset", "audio_path", "genre_top", "genre_id", "genre_title", "exists"]
    return manifest[cols].sort_values("track_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-csv", type=Path, required=True)
    parser.add_argument("--genres-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=["small", "medium", "large"],
        default="small",
        help="FMA subset to build. medium includes small+medium; large includes all three.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.tracks_csv, args.genres_csv, args.audio_root, args.subset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    total = len(manifest)
    found = int(manifest["exists"].sum())
    print(f"Wrote {args.output}")
    print(f"Tracks in FMA {args.subset} metadata: {total}")
    print(f"Audio files found: {found}")
    print(f"Missing audio files: {total - found}")
    print("Genre counts:")
    print(manifest["genre_top"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
