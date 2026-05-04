"""
extract_cnn_embeddings.py

Extracts 256-dim CNN embeddings from trained model.
Usage:
    python src/extract_cnn_embeddings.py --debug
    python src/extract_cnn_embeddings.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from cnn_model import MusicCNN
from train_cnn import FMADataset, audio_to_melspec

# ── Config ───────────────────────────────────────────────────────────────────
MANIFEST_PATH  = "data/processed/fma_small_manifest.csv"
LABELS_PATH    = "data/processed/track_labels.csv"
MODEL_PATH     = "data/processed/cnn_model.pth"
ENCODER_PATH   = "data/processed/label_encoder.npy"
OUTPUT_FULL    = "data/processed/cnn_embeddings.csv"
OUTPUT_DEBUG   = "data/processed/cnn_embeddings_debug.csv"
FAILURES_PATH  = "data/processed/cnn_embeddings_failures.csv"
DEBUG_N        = 25
BATCH_SIZE     = 32
# ─────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main(debug: bool):
    os.makedirs("data/processed", exist_ok=True)
    device = get_device()

    # Load manifest and labels
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["exists"] == True].reset_index(drop=True)
    labels   = pd.read_csv(LABELS_PATH)
    manifest = manifest[manifest["track_id"].isin(labels["track_id"])]

    if debug:
        manifest = manifest.head(DEBUG_N)
        print(f"Debug mode: {DEBUG_N} tracks")

    # Load encoder
    classes = np.load(ENCODER_PATH, allow_pickle=True)
    encoder = LabelEncoder()
    encoder.classes_ = classes

    # Load trained model
    model = MusicCNN(num_genres=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    dataset = FMADataset(manifest, labels, encoder)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

    results  = []
    failures = []
    total    = len(dataset)

    with torch.no_grad():
        for mels, label_idxs, track_ids in loader:
            try:
                mels = mels.to(device)
                embs = model(mels, return_embedding=True)  # (batch, 256)
                embs = embs.cpu().numpy()

                for i, track_id in enumerate(track_ids):
                    row_dict = {"track_id": int(track_id)}
                    for j, val in enumerate(embs[i]):
                        row_dict[f"emb_{j:03d}"] = val
                    results.append(row_dict)

                if len(results) % 100 == 0 or len(results) <= BATCH_SIZE:
                    print(f"[{len(results)}/{total}] Extracted embeddings")

            except Exception as e:
                for track_id in track_ids:
                    failures.append({"track_id": int(track_id), "error": str(e)})
                print(f"  FAILED batch: {e}")

    # Save
    output_path = OUTPUT_DEBUG if debug else OUTPUT_FULL
    if results:
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\nSaved {len(results)} embeddings → {output_path}")

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        print(f"Logged {len(failures)} failures → {FAILURES_PATH}")

    print(f"\nDone! Success: {len(results)}, Failed: {len(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(debug=args.debug)