"""
src/predict.py

End-to-end inference on a single new document image:
  1. Preprocess (denoise / threshold / deskew)
  2. Run OCR to extract text
  3. Run the CNN to classify the document type
  4. Return everything as one structured result

Run with:  python -m src.predict path/to/new_document.jpg
"""

import sys
import os

import torch
import cv2
from PIL import Image

import config
from src.preprocess import preprocess_image
from src.ocr import extract_text_with_confidence
from src.model import build_model, get_device
from src.dataset import get_transforms


_model_cache = {}


def load_model():
    """Loads the trained model once and caches it for repeat calls (e.g. in app.py)."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["classes"], _model_cache["device"]

    device = get_device()

    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_SAVE_PATH}. "
            f"Run `python -m src.train` first."
        )

    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    model = build_model(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _model_cache["model"] = model
    _model_cache["classes"] = checkpoint["classes"]
    _model_cache["device"] = device

    return model, checkpoint["classes"], device


@torch.no_grad()
def classify_image(image_path: str) -> dict:
    model, classes, device = load_model()

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    predicted_idx = int(torch.argmax(probs).item())

    return {
        "predicted_class": classes[predicted_idx],
        "confidence": round(float(probs[predicted_idx]) * 100, 2),
        "all_class_probs": {classes[i]: round(float(probs[i]) * 100, 2) for i in range(len(classes))},
    }


def process_document(image_path: str) -> dict:
    """
    Full pipeline: preprocess (for CNN/visual reference) -> OCR -> classify -> combined result.

    NOTE: OCR runs on the ORIGINAL image, not the heavily denoised/thresholded
    'cleaned' version. Our fixed-blocksize adaptive threshold is tuned for
    scanned forms with large, sparse text -- on dense, small-font documents
    like resumes it erases fine detail and OCR quality gets much worse.
    Tesseract has its own internal binarization that generally outperforms a
    one-size-fits-all threshold for this kind of document.
    """
    _ = preprocess_image(image_path)  # kept for the app's side-by-side preview
    ocr_result = extract_text_with_confidence(image_path)
    classification_result = classify_image(image_path)

    return {
        "file": image_path,
        "predicted_class": classification_result["predicted_class"],
        "classification_confidence": classification_result["confidence"],
        "class_probabilities": classification_result["all_class_probs"],
        "ocr_text": ocr_result["text"],
        "ocr_avg_confidence": ocr_result["avg_confidence"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <image_path>")
        sys.exit(1)

    result = process_document(sys.argv[1])

    print(f"\nFile: {result['file']}")
    print(f"Predicted class: {result['predicted_class']} ({result['classification_confidence']}% confidence)")
    print(f"Class probabilities: {result['class_probabilities']}")
    print(f"OCR confidence: {result['ocr_avg_confidence']}%")
    print("---- Extracted Text ----")
    print(result["ocr_text"][:500])
