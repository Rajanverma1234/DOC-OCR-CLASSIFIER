"""
Central configuration for the Document OCR & Classification project.
Change values here instead of hunting through every script.
"""

import os

# ---------------------------------------------------------------------
# PATHS  (CHANGE THESE to match your machine / dataset location)
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")          # put your raw scanned images here, one sub-folder per class
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")  # cleaned images get saved here automatically
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models", "doc_classifier.pt")
OCR_OUTPUT_DIR = os.path.join(BASE_DIR, "sample_output")

# ---------------------------------------------------------------------
# CLASSES  <-- CHANGE THIS to match your own document categories
# ---------------------------------------------------------------------
# The folder names inside data/raw/ MUST exactly match this list.
# Example layout:
#   data/raw/invoice/*.jpg
#   data/raw/resume/*.jpg
#   data/raw/id_card/*.jpg
#   data/raw/receipt/*.jpg
CLASSES = ["invoice", "resume", "id_card", "receipt"]
NUM_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------
# TRAINING HYPERPARAMETERS  <-- tune these as needed
# ---------------------------------------------------------------------
IMAGE_SIZE = 224          # ResNet expects 224x224
BATCH_SIZE = 16           # lower this if you run out of memory
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2           # 20% of data held out for validation
RANDOM_SEED = 42

# ---------------------------------------------------------------------
# OCR (Tesseract) SETTINGS
# ---------------------------------------------------------------------
# On Windows, uncomment and set the path to your Tesseract install, e.g.:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None   # leave as None on Linux/Mac if tesseract is on your PATH

OCR_LANG = "eng"

# ---------------------------------------------------------------------
# DEVICE
# ---------------------------------------------------------------------
DEVICE = "cuda"  # will auto-fallback to "cpu" in code if CUDA isn't available
