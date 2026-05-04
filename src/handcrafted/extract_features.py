#!/usr/bin/env python3
"""Extract handcrafted audio features for the music similarity project."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def summarize(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def extract_features(audio_path: Path, sr: int, duration: float) -> dict[str, float]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    if len(y) == 0:
        raise ValueError("empty audio")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    rms_mean, rms_std = summarize(librosa.feature.rms(y=y)[0])
    zcr_mean, zcr_std = summarize(librosa.feature.zero_crossing_rate(y)[0])
    centroid_mean, centroid_std = summarize(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    bandwidth_mean, bandwidth_std = summarize(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
    rolloff_mean, rolloff_std = summarize(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_stds = np.std(mfcc, axis=1)

    row = {
        "tempo": tempo,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "spectral_centroid_mean": centroid_mean,
        "spectral_centroid_std": centroid_std,
        "spectral_bandwidth_mean": bandwidth_mean,
        "spectral_bandwidth_std": bandwidth_std,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_rolloff_std": rolloff_std,
    }

    for idx, value in enumerate(mfcc_means, start=1):
        row[f"mfcc_{idx:02d}_mean"] = float(value)
    for idx, value in enumerate(mfcc_stds, start=1):
        row[f"mfcc_{idx:02d}_std"] = float(value)

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional small run for debugging, for example --limit 25.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["exists"]].copy()

    if args.limit is not None:
        manifest = manifest.head(args.limit)

    rows = []
    failures = []

    for item in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        try:
            features = extract_features(Path(item.audio_path), args.sr, args.duration)
            rows.append({"track_id": item.track_id, **features})
        except Exception as exc:  # noqa: BLE001 - record bad files and continue.
            failures.append({"track_id": item.track_id, "error": str(exc)})

    features_df = pd.DataFrame(rows).sort_values("track_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(features_df)} tracks")

    if failures:
        failures_path = args.output.with_name(args.output.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Wrote failures to {failures_path}")


if __name__ == "__main__":
    main()
