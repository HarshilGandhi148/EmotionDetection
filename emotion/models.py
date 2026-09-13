"""Small, randomly initialized CNNs. Both return unnormalized class logits."""
import torch
from torch import nn

from emotion import CLASSES


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, len(CLASSES)),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        channels = 1
        for width in (32, 64, 128):
            layers.extend([
                nn.Conv2d(channels, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2),
            ])
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(128, len(CLASSES)),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_model(name):
    if name not in ("baseline", "cnn"):
        raise ValueError(f"Unknown architecture: {name}")
    return BaselineCNN() if name == "baseline" else EmotionCNN()


def select_device(name="auto"):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; use --device cpu.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    return torch.device(name)
