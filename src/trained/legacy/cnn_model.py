"""
cnn_model.py
CNN architecture for music genre classification.
Penultimate layer produces 256-dim embeddings.
"""

import torch
import torch.nn as nn


class MusicCNN(nn.Module):
    def __init__(self, num_genres=8):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Linear(256, num_genres)

    def forward(self, x, return_embedding=False):
        x = self.features(x)
        emb = self.embedding(x)        # 256-dim embedding
        if return_embedding:
            return emb
        return self.classifier(emb)