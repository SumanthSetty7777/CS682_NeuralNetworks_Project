"""
extract_pretrained_embeddings.py

Extracts VGGish pretrained audio embeddings from FMA small dataset.
Usage:
    python src/extract_pretrained_embeddings.py              # full run (8000 tracks)
    python src/extract_pretrained_embeddings.py --debug      # debug run (25 tracks)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torchvggish import vggish, vggish_input

# ── Config ──────────────────────────────────────────────────────────────────
MANIFEST_PATH  = "data/processed/fma_small_manifest.csv"
OUTPUT_FULL    = "data/processed/pretrained_embeddings.csv"
OUTPUT_DEBUG   = "data/processed/pretrained_embeddings_debug.csv"
FAILURES_PATH  = "data/processed/pretrained_embeddings_failures.csv"
DEBUG_N        = 25          # number of tracks to process in debug mode
SAMPLE_RATE    = 16000       # VGGish requires 16kHz
EMBEDDING_SIZE = 128         # VGGish output dimension
# ────────────────────────────────────────────────────────────────────────────


def load_manifest(debug: bool) -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["exists"] == True].reset_index(drop=True)
    print(f"Total tracks available: {len(manifest)}")
    if debug:
        manifest = manifest.head(DEBUG_N)
        print(f"Debug mode: processing first {DEBUG_N} tracks")
    return manifest


def load_model():
    model = vggish()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Running on GPU")
    else:
        print("Running on CPU")
    return model


def extract_embedding(audio_path: str, model) -> np.ndarray:
    audio, sr = sf.read(audio_path, always_2d=False)

    # Convert stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        import resampy
        audio = resampy.resample(audio, sr, SAMPLE_RATE)

    # Convert raw audio → mel spectrogram frames
    # waveform_to_examples already returns (n_frames, 1, 96, 64)
    examples = vggish_input.waveform_to_examples(audio, SAMPLE_RATE)
    examples = torch.tensor(examples).float()  # ← NO unsqueeze needed

    if torch.cuda.is_available():
        examples = examples.cuda()

    with torch.no_grad():
        emb = model.forward(examples)

    track_embedding = emb.mean(axis=0).cpu().numpy()  # shape: (128,)
    return track_embedding


def main(debug: bool):
    os.makedirs("data/processed", exist_ok=True)

    manifest = load_manifest(debug)
    output_path = OUTPUT_DEBUG if debug else OUTPUT_FULL
    model = load_model()

    results  = []   # successful embeddings
    failures = []   # tracks that failed

    total = len(manifest)
    for i, row in manifest.iterrows():
        track_id   = row["track_id"]
        audio_path = row["audio_path"]

        try:
            embedding = extract_embedding(audio_path, model)

            # Build one row: track_id + emb_000 ... emb_127
            row_dict = {"track_id": track_id}
            for j, val in enumerate(embedding):
                row_dict[f"emb_{j:03d}"] = val

            results.append(row_dict)

            # Progress update every 10 tracks
            if (len(results)) % 10 == 0 or len(results) == 1:
                print(f"[{len(results)}/{total}] Processed track {track_id}")

        except Exception as e:
            print(f"  FAILED track {track_id}: {e}")
            failures.append({"track_id": track_id, "error": str(e)})

    # ── Save results ─────────────────────────────────────────────────────
    if results:
        df_out = pd.DataFrame(results)
        df_out.to_csv(output_path, index=False)
        print(f"\nSaved {len(results)} embeddings → {output_path}")
    else:
        print("\nNo embeddings were successfully extracted.")

    if failures:
        df_fail = pd.DataFrame(failures)
        df_fail.to_csv(FAILURES_PATH, index=False)
        print(f"Logged {len(failures)} failures → {FAILURES_PATH}")

    print(f"\nDone! Success: {len(results)}, Failed: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VGGish pretrained embeddings")
    parser.add_argument("--debug", action="store_true",
                        help=f"Run on first {DEBUG_N} tracks only")
    args = parser.parse_args()
    main(debug=args.debug)