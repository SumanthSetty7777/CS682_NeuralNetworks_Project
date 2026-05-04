"""
train_cnn.py

Trains CNN on FMA small mel spectrograms for genre classification.
Usage:
    python src/train_cnn.py --debug     # 25 tracks, 2 epochs
    python src/train_cnn.py             # full run
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from cnn_model import MusicCNN

# ── Config ───────────────────────────────────────────────────────────────────
MANIFEST_PATH = "data/processed/fma_small_manifest.csv"
LABELS_PATH   = "data/processed/track_labels.csv"
MODEL_PATH    = "data/processed/cnn_model.pth"
ENCODER_PATH  = "data/processed/label_encoder.npy"
DEBUG_N       = 25
SAMPLE_RATE   = 22050
DURATION      = 10        # seconds per clip
N_MELS        = 128
HOP_LENGTH    = 512
N_FFT         = 2048
BATCH_SIZE    = 32
EPOCHS        = 10
LEARNING_RATE = 0.001
# ─────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def audio_to_melspec(audio_path: str) -> np.ndarray:
    """Load audio and convert to mel spectrogram (128x128)."""
    audio, sr = sf.read(audio_path, always_2d=False)

    # Stereo to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)

    # Resample if needed
    if sr != SAMPLE_RATE:
        import resampy
        audio = resampy.resample(audio, sr, SAMPLE_RATE)

    # Trim or pad to fixed duration
    target_len = SAMPLE_RATE * DURATION
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize to 128x128
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)
    return mel_db.astype(np.float32)


class FMADataset(Dataset):
    def __init__(self, manifest, labels, encoder):
        self.manifest = manifest.reset_index(drop=True)
        self.labels   = labels
        self.encoder  = encoder

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row        = self.manifest.iloc[idx]
        audio_path = row["audio_path"]
        track_id   = row["track_id"]
        label      = self.labels[self.labels["track_id"] == track_id]["genre"].values[0]
        label_idx  = self.encoder.transform([label])[0]

        mel = audio_to_melspec(audio_path)
        mel = torch.tensor(mel).unsqueeze(0)  # (1, 128, 128)
        return mel, label_idx, track_id


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for mel, labels, _ in loader:
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(mel)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for mel, labels, _ in loader:
            mel, labels = mel.to(device), labels.to(device)
            out         = model(mel)
            loss        = criterion(out, labels)
            total_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def main(debug: bool):
    os.makedirs("data/processed", exist_ok=True)
    device = get_device()

    # Load manifest and labels
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest = manifest[manifest["exists"] == True].reset_index(drop=True)
    labels   = pd.read_csv(LABELS_PATH)

    # Merge to keep only tracks with labels
    manifest = manifest[manifest["track_id"].isin(labels["track_id"])]

    if debug:
        manifest = manifest.head(DEBUG_N)
        print(f"Debug mode: {DEBUG_N} tracks, 2 epochs")

    # Encode genre labels
    encoder = LabelEncoder()
    encoder.fit(labels["genre"])
    np.save(ENCODER_PATH, encoder.classes_)
    print(f"Genres: {list(encoder.classes_)}")

    # Train/val split
    train_df, val_df = train_test_split(manifest, test_size=0.2, random_state=42)

    train_dataset = FMADataset(train_df, labels, encoder)
    val_dataset   = FMADataset(val_df,   labels, encoder)

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model     = MusicCNN(num_genres=len(encoder.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    epochs = 2 if debug else EPOCHS
    best_val_acc = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.3f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.3f} Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model (val acc: {val_acc:.3f})")

    print(f"\nTraining done! Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(debug=args.debug)