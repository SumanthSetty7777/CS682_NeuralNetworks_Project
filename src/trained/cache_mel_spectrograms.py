#!/usr/bin/env python3
"""Cache fixed-length mel spectrograms for CNN training."""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_excerpt(audio_path: Path, sr: int, duration: float, offset: float) -> np.ndarray:
    target_samples = int(round(sr * duration))
    y, _ = librosa.load(audio_path, sr=sr, mono=True, offset=offset, duration=duration)

    if len(y) < sr:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=duration)

    if len(y) < target_samples:
        y = np.pad(y, (0, target_samples - len(y)))

    return y[:target_samples]


def make_mel(
    audio_path: Path,
    sr: int,
    duration: float,
    offset: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    y = load_excerpt(audio_path, sr=sr, duration=duration, offset=offset)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_norm.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--offset", type=float, default=7.5)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-per-genre",
        type=int,
        default=None,
        help="Optional balanced debug run, for example --limit-per-genre 8.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["exists"]].copy()
    manifest = manifest.dropna(subset=["genre_top"]).sort_values("track_id")

    if args.limit_per_genre is not None:
        manifest = manifest.groupby("genre_top", group_keys=False).head(args.limit_per_genre)

    if args.limit is not None:
        manifest = manifest.head(args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.index_output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for item in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        track_id = int(item.track_id)
        mel_path = args.output_dir / f"{track_id:06d}.npy"

        try:
            if args.overwrite or not mel_path.exists():
                mel = make_mel(
                    audio_path=Path(item.audio_path),
                    sr=args.sr,
                    duration=args.duration,
                    offset=args.offset,
                    n_mels=args.n_mels,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                )
                np.save(mel_path, mel)

            rows.append(
                {
                    "track_id": track_id,
                    "mel_path": str(mel_path),
                    "genre_top": item.genre_top,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep processing other tracks.
            failures.append({"track_id": track_id, "error": str(exc)})

    pd.DataFrame(rows).to_csv(args.index_output, index=False)
    print(f"Wrote {args.index_output} with {len(rows)} tracks")

    if failures:
        failures_path = args.index_output.with_name(args.index_output.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Wrote failures to {failures_path}")


if __name__ == "__main__":
    main()
