"""
src/train.py

Trains the document classifier and saves the best checkpoint
(by validation accuracy) to config.MODEL_SAVE_PATH.

Run with:  python -m src.train      (from the project root)
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

import config
from src.dataset import get_dataloaders
from src.model import build_model, get_device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    device = get_device()
    print(f"Training on device: {device}")

    train_loader, val_loader = get_dataloaders()
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)}")

    model = build_model(freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE)

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": config.CLASSES,
                    "val_acc": val_acc,
                },
                config.MODEL_SAVE_PATH,
            )
            print(f"  -> New best model saved (val_acc={val_acc:.4f}) to {config.MODEL_SAVE_PATH}")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
