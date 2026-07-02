"""
src/ocr.py

Extracts text from a (preferably preprocessed) document image using Tesseract OCR.
Requires the Tesseract binary installed on your system, separate from pytesseract
the Python wrapper. See README.md -> "Installing Tesseract".
"""

import cv2
import pytesseract
import numpy as np

import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def extract_text(image: np.ndarray | str) -> str:
    """
    Accepts either a numpy array (already loaded/preprocessed image)
    or a file path string, and returns the extracted text.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not read image: {image}")

    text = pytesseract.image_to_string(image, lang=config.OCR_LANG)
    return text.strip()


def extract_text_with_confidence(image: np.ndarray | str) -> dict:
    """
    Returns text plus per-word confidence scores and average confidence.
    Useful for flagging low-quality scans that may need manual review.
    """
    if isinstance(image, str):
        image = cv2.imread(image)
        if image is None:
            raise ValueError(f"Could not read image: {image}")

    data = pytesseract.image_to_data(image, lang=config.OCR_LANG, output_type=pytesseract.Output.DICT)

    words = [w for w in data["text"] if w.strip()]
    confidences = [int(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and int(c) >= 0]

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": " ".join(words),
        "avg_confidence": round(avg_confidence, 2),
        "word_count": len(words),
    }


if __name__ == "__main__":
    # Quick manual test: python src/ocr.py path/to/image.jpg
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/ocr.py <image_path>")
        sys.exit(1)

    result = extract_text_with_confidence(sys.argv[1])
    print(f"Average OCR confidence: {result['avg_confidence']}%")
    print(f"Word count: {result['word_count']}")
    print("---- Extracted Text ----")
    print(result["text"])
