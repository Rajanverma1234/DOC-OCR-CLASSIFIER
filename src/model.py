"""
src/model.py

CNN document classifier built on a pretrained ResNet18 backbone
(transfer learning). Only the final classification layer is trained
from scratch initially -- this is what lets the model reach good
accuracy even with a fairly small labeled dataset.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


def build_model(num_classes: int = None, freeze_backbone: bool = True) -> nn.Module:
    """
    Loads a pretrained ResNet18 and replaces its final layer to output
    `num_classes` scores instead of the original 1000 ImageNet classes.

    freeze_backbone=True freezes all convolutional layers so only the new
    final layer trains -- fast, and works well with small datasets.
    Set to False for full fine-tuning once you have more data.
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer -- this part always trains,
    # regardless of freeze_backbone, since it's created fresh.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_backbone(model: nn.Module):
    """
    Call this after a few epochs of training the head only, to fine-tune
    the whole network at a lower learning rate. Improves accuracy once
    the classifier head has stabilized.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_device() -> torch.device:
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    # Quick sanity check: build the model and run one dummy batch through it
    device = get_device()
    print(f"Using device: {device}")

    model = build_model().to(device)
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}  (expected: [2, {config.NUM_CLASSES}])")
