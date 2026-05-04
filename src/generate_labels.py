"""
generate_labels.py

Generates track_labels.csv from FMA metadata tracks.csv
Usage:
    python src/generate_labels.py
"""

import pandas as pd
import os

# ── Config ──────────────────────────────────────────────────────────────────
TRACKS_PATH = "data/raw/fma_metadata/tracks.csv"   # update if needed
OUTPUT_PATH = "data/processed/track_labels.csv"
# ────────────────────────────────────────────────────────────────────────────


def find_tracks_csv():
    """Try common locations for tracks.csv if default path fails."""
    common_paths = [
        "data/raw/fma_metadata/tracks.csv",
        "fma_metadata/tracks.csv",
        "../fma_metadata/tracks.csv",
        os.path.expanduser("~/Downloads/fma_metadata/tracks.csv"),
        os.path.expanduser("~/fma_metadata/tracks.csv"),
    ]
    for path in common_paths:
        if os.path.exists(path):
            print(f"Found tracks.csv at: {path}")
            return path
    return None


def main():
    os.makedirs("data/processed", exist_ok=True)

    # Auto-find tracks.csv
    tracks_path = find_tracks_csv()
    if tracks_path is None:
        print("ERROR: tracks.csv not found!")
        print("Please find it on your system and update TRACKS_PATH in this script")
        print("Run: find / -name tracks.csv 2>/dev/null")
        return

    # Load tracks metadata
    print(f"Loading tracks from {tracks_path}...")
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])

    # Extract track_id and top-level genre
    labels = tracks["track"]["genre_top"].dropna().reset_index()
    labels.columns = ["track_id", "genre"]
    print(f"Total tracks with genre labels: {len(labels)}")

    # Keep only top 8 genres (fma_small genres)
    top_genres = labels["genre"].value_counts().head(8).index.tolist()
    labels = labels[labels["genre"].isin(top_genres)]
    print(f"Tracks after filtering to top 8 genres: {len(labels)}")

    # Save
    labels.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(labels)} track labels → {OUTPUT_PATH}")
    print(f"\nGenre distribution:")
    print(labels["genre"].value_counts())


if __name__ == "__main__":
    main()