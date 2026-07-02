"""
src/evaluate.py

Loads a trained checkpoint and reports precision/recall/F1 per class,
plus a confusion matrix. Run after training to sanity-check performance
before deploying.

Run with:  python -m src.evaluate      (from the project root)
"""

import torch
from sklearn.metrics import classification_report, confusion_matrix

import config
from src.dataset import get_dataloaders
from src.model import build_model, get_device


@torch.no_grad()
def run_evaluation():
    device = get_device()

    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model = build_model(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint with validation accuracy: {checkpoint['val_acc']:.4f}")

    _, val_loader = get_dataloaders()

    all_preds, all_labels = [], []
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=checkpoint["classes"], zero_division=0))

    print("Confusion Matrix (rows = actual, cols = predicted):")
    print(checkpoint["classes"])
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    run_evaluation()
