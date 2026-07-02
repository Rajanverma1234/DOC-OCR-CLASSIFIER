"""
src/preprocess.py

Cleans up noisy scanned document images before they go to OCR or the CNN.
Pipeline: grayscale -> denoise -> adaptive threshold -> deskew.
"""

import os
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray) -> np.ndarray:
    """Removes salt-and-pepper / scan noise while keeping text edges sharp."""
    return cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)


def adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Binarizes the image so faint or unevenly-lit text becomes crisp black/white."""
    return cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detects and corrects small rotation angles caused by crooked scanning.
    Works on a binarized (black text / white background) image.
    """
    coords = np.column_stack(np.where(image < 255))
    if coords.shape[0] == 0:
        return image  # blank image, nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]

    # cv2.minAreaRect returns angles in the range [-90, 0); normalize it
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Full pipeline for one image path -> cleaned, deskewed, binarized numpy array.
    Raises FileNotFoundError / ValueError on bad input so callers can catch it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image (unsupported format?): {image_path}")

    gray = to_grayscale(image)
    clean = denoise(gray)
    binary = adaptive_threshold(clean)
    straightened = deskew(binary)
    return straightened


def preprocess_and_save(image_path: str, output_dir: str) -> str:
    """Runs preprocess_image() and writes the result to output_dir. Returns the new path."""
    os.makedirs(output_dir, exist_ok=True)
    processed = preprocess_image(image_path)
    filename = os.path.basename(image_path)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, processed)
    return out_path


if __name__ == "__main__":
    # Quick manual test: python src/preprocess.py path/to/image.jpg
    import sys
    if len(sys.argv) != 2:
        print("Usage: python src/preprocess.py <image_path>")
        sys.exit(1)

    out = preprocess_and_save(sys.argv[1], output_dir="data/processed")
    print(f"Saved cleaned image to: {out}")
