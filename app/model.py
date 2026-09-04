from pathlib import Path

import torch
import torch.nn as nn


CLASS_NAMES = ["benign", "malignant", "normal"]


class BUSICNN(nn.Module):
    """The same CNN architecture used by the BUSI training project."""

    def __init__(self, number_of_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, number_of_classes),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def load_model(model_path):
    """Load trained BUSI CNN weights."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found:\n{model_path}\n\n"
            "Copy your trained .pth file into the project folder."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BUSICNN(number_of_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, device
