#!/usr/bin/env python3
"""Train a CNN genre classifier and export penultimate embeddings."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset


class MelDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: np.ndarray | None = None) -> None:
        self.frame = frame.reset_index(drop=True)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.frame.iloc[index]
        mel = np.load(row["mel_path"]).astype(np.float32)
        x = torch.from_numpy(mel).unsqueeze(0)

        if self.labels is None:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(int(self.labels[index]), dtype=torch.long)

        return x, y, int(row["track_id"])


def conv_stage(in_channels: int, out_channels: int, repeats: int) -> nn.Sequential:
    layers = []
    for idx in range(repeats):
        layers.extend(
            [
                nn.Conv2d(
                    in_channels if idx == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        )
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


def residual_stage(
    in_channels: int,
    out_channels: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
    layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
    return nn.Sequential(*layers)


class GenreCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_size: str = "small",
        embedding_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if model_size == "resnet":
            channels = [32, 64, 128, 256]
            self.features = nn.Sequential(
                nn.Conv2d(1, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                residual_stage(channels[0], channels[0], blocks=2, stride=1),
                residual_stage(channels[0], channels[1], blocks=2, stride=2),
                residual_stage(channels[1], channels[2], blocks=2, stride=2),
                residual_stage(channels[2], channels[3], blocks=2, stride=2),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif model_size == "small":
            channels = [16, 32, 64, 128]
            repeats = [1, 1, 1, 1]
        elif model_size == "medium":
            channels = [32, 64, 128, 256]
            repeats = [2, 2, 2, 1]
        else:
            raise ValueError(f"Unknown model_size: {model_size}")

        if model_size in {"small", "medium"}:
            stages = []
            in_channels = 1
            for out_channels, repeat_count in zip(channels, repeats, strict=True):
                stages.append(conv_stage(in_channels, out_channels, repeat_count))
                in_channels = out_channels
            stages.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.features = nn.Sequential(*stages)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, apply_dropout=True))

    def embed(self, x: torch.Tensor, apply_dropout: bool = False) -> torch.Tensor:
        x = self.features(x)
        if apply_dropout:
            return self.embedding(x)

        training = self.embedding.training
        self.embedding.eval()
        with torch.no_grad():
            embedding = self.embedding(x)
        self.embedding.train(training)
        return embedding


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_items = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(x)
        correct += int((logits.argmax(dim=1) == y).sum().item())
        total_items += len(x)

    return total_loss / total_items, correct / total_items


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_items = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += float(loss.item()) * len(x)
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total_items += len(x)

    return total_loss / total_items, correct / total_items


def export_embeddings(
    model: GenreCNN,
    frame: pd.DataFrame,
    output: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    loader = DataLoader(MelDataset(frame), batch_size=batch_size, shuffle=False)
    model.eval()

    rows = []
    with torch.no_grad():
        for x, _, track_ids in loader:
            embeddings = model.embed(x.to(device)).cpu().numpy()
            for track_id, embedding in zip(track_ids.numpy(), embeddings, strict=True):
                row = {"track_id": int(track_id)}
                row.update({f"emb_{idx:03d}": float(value) for idx, value in enumerate(embedding)})
                rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output} with {len(rows)} embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mel-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embeddings-output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-size", choices=["small", "medium", "resnet"], default="small")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop early after this many epochs without validation accuracy improvement.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device()
    print(f"Using device: {device}")

    data = pd.read_csv(args.mel_index).sort_values("track_id").reset_index(drop=True)
    if args.limit is not None:
        data = data.head(args.limit).copy()

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data["genre_top"])

    train_val_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )
    val_fraction = args.val_size / (1.0 - args.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=args.seed,
        stratify=labels[train_val_idx],
    )

    train_data = data.iloc[train_idx].reset_index(drop=True)
    val_data = data.iloc[val_idx].reset_index(drop=True)
    test_data = data.iloc[test_idx].reset_index(drop=True)
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    train_loader = DataLoader(
        MelDataset(train_data, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MelDataset(val_data, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MelDataset(test_data, test_labels),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = GenreCNN(
        num_classes=len(label_encoder.classes_),
        model_size=args.model_size,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0
    best_model_path = args.output_dir / "cnn_genre_best.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"train_accuracy={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience is not None and epochs_without_improvement >= args.patience:
            print(f"Stopping early after {args.patience} epochs without validation improvement.")
            break

    history_path = args.output_dir / "cnn_genre_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Wrote {history_path}")

    mapping_path = args.output_dir / "cnn_genre_labels.json"
    mapping_path.write_text(
        json.dumps({int(idx): label for idx, label in enumerate(label_encoder.classes_)}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {mapping_path}")

    split_frame = data[["track_id", "genre_top"]].copy()
    split_frame["split"] = "train"
    split_frame.loc[val_idx, "split"] = "val"
    split_frame.loc[test_idx, "split"] = "test"
    split_path = args.output_dir / "cnn_genre_split.csv"
    split_frame.to_csv(split_path, index=False)
    print(f"Wrote {split_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    metrics_path = args.output_dir / "cnn_genre_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_accuracy": best_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "train_tracks": int(len(train_data)),
                "val_tracks": int(len(val_data)),
                "test_tracks": int(len(test_data)),
                "model_size": args.model_size,
                "embedding_dim": args.embedding_dim,
                "dropout": args.dropout,
                "label_smoothing": args.label_smoothing,
                "epochs_run": int(len(history)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {metrics_path}")
    export_embeddings(model, data, args.embeddings_output, args.batch_size, device)
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
