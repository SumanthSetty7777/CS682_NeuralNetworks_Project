#!/usr/bin/env python3
"""Extract VGGish pretrained audio embeddings from an FMA manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path(".torch_cache").resolve()))

import librosa
import numpy as np
import pandas as pd
import torch
from torchvggish import vggish, vggish_input
from tqdm import tqdm


SAMPLE_RATE = 16000
DEFAULT_DEBUG_N = 25


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_manifest(manifest_path: Path, debug: bool, debug_n: int) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["exists"]].copy()
    manifest = manifest.sort_values("track_id").reset_index(drop=True)
    if debug:
        manifest = manifest.head(debug_n)
    return manifest


def load_model(device: torch.device) -> torch.nn.Module:
    model = vggish()
    model.eval()
    model = model.to(device)
    return model


def extract_embedding(audio_path: Path, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)

    examples = vggish_input.waveform_to_examples(audio, SAMPLE_RATE)
    examples_tensor = torch.as_tensor(examples, dtype=torch.float32, device=device)

    with torch.no_grad():
        embeddings = model(examples_tensor)

    return embeddings.mean(dim=0).detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/fma_small_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/pretrained_embeddings.csv"))
    parser.add_argument("--failures-output", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-n", type=int, default=DEFAULT_DEBUG_N)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if args.debug:
        output = output.with_name(output.stem + "_debug" + output.suffix)

    failures_output = args.failures_output
    if failures_output is None:
        failures_output = output.with_name(output.stem + "_failures.csv")

    manifest = load_manifest(args.manifest, args.debug, args.debug_n)
    print(f"Tracks to process: {len(manifest)}")

    device = select_device(args.device)
    try:
        model = load_model(device)
    except RuntimeError:
        if args.device != "auto" or device.type == "cpu":
            raise
        print(f"Could not initialize VGGish on {device}; falling back to cpu.")
        device = torch.device("cpu")
        model = load_model(device)
    print(f"Using device: {device}")

    rows = []
    failures = []
    for item in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        try:
            embedding = extract_embedding(Path(item.audio_path), model, device)
            row = {"track_id": int(item.track_id)}
            row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding)})
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - record failed files and keep going.
            failures.append({"track_id": int(item.track_id), "error": str(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output} with {len(rows)} embeddings")

    if failures:
        failures_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failures_output, index=False)
        print(f"Wrote {failures_output} with {len(failures)} failures")


if __name__ == "__main__":
    main()
