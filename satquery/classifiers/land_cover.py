"""
Land-cover classification model: ResNet-18 + multi-label head.

Replaces the ImageNet fc layer with a 19-class sigmoid head matching the
CORINE-derived BigEarthNet-S2 label scheme.  Checkpoint loading from
``models/land_cover/`` is supported via :func:`load_from_checkpoint`.

Architecture
------------
- Backbone: ``torchvision.models.resnet18`` (ImageNet-pretrained)
- Head: ``nn.Linear(512, 19)``  — no final activation; use
  ``BCEWithLogitsLoss`` during training and ``torch.sigmoid`` at inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# ---------------------------------------------------------------------------
# BigEarthNet 19-class CORINE-derived label set
# ---------------------------------------------------------------------------
BIGEARTHNET_19_CLASSES: list[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

NUM_CLASSES: int = len(BIGEARTHNET_19_CLASSES)


# ---------------------------------------------------------------------------
# Transforms (ImageNet normalisation — matches the pretrained backbone)
# ---------------------------------------------------------------------------
def get_transforms(train: bool = False) -> transforms.Compose:
    """Return preprocessing transforms for the ResNet-18 backbone.

    Parameters
    ----------
    train : bool
        If ``True``, adds random horizontal/vertical flips for data
        augmentation during training.
    """
    ops: list[transforms.transforms._BasicTransform | nn.Module] = [
        transforms.Resize((224, 224)),
    ]
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LandCoverModel(nn.Module):
    """ResNet-18 backbone with a multi-label classification head.

    Parameters
    ----------
    num_classes : int
        Number of output labels (default: 19 BigEarthNet classes).
    pretrained_backbone : bool
        Load ImageNet-pretrained weights for the ResNet-18 backbone.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.classes = BIGEARTHNET_19_CLASSES

        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)

        # Keep everything except the original fc layer.
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 512, 1, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits ``(B, num_classes)``.

        Apply ``torch.sigmoid`` for probabilities at inference time.
        """
        h = self.features(x)           # (B, 512, 1, 1)
        h = h.view(h.size(0), -1)      # (B, 512)
        return self.fc(h)               # (B, num_classes)

    # ------------------------------------------------------------------
    # Convenience I/O
    # ------------------------------------------------------------------
    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "LandCoverModel":
        """Instantiate a model and load fine-tuned weights.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to a ``.pt`` or ``.pth`` file saved by the training script.
        device : str | None
            Target device (``"cpu"``, ``"cuda"``).  Auto-detected when *None*.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls(pretrained_backbone=False)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Support both bare state-dicts and dicts wrapped as {"model_state_dict": ...}
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
