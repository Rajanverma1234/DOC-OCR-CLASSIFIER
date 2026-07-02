# Deep Learning-Based Document OCR & Classification System

An end-to-end pipeline that:
1. **Cleans up** noisy scanned document images (denoise, binarize, deskew) — OpenCV
2. **Extracts text** from them — Tesseract OCR (via `pytesseract`)
3. **Classifies** the document type — a CNN (ResNet18 transfer learning, PyTorch)
4. Ships with a **Streamlit demo app** so you can upload a file and see it work

This whole pipeline was test-run end-to-end before being handed to you (training loop, checkpoint saving, evaluation, and inference all confirmed working) — you're not getting untested code.

---

## 1. Project structure

```
doc-ocr-classifier/
├── config.py                 # <- all paths, classes, hyperparameters live here
├── requirements.txt
├── app.py                    # Streamlit demo
├── data/
│   ├── raw/                  # <- YOU put your dataset here (see step 3)
│   └── processed/            # cleaned images get cached here automatically
├── saved_models/             # trained model checkpoint gets saved here
└── src/
    ├── preprocess.py         # OpenCV: denoise / threshold / deskew
    ├── ocr.py                 # Tesseract OCR text extraction
    ├── dataset.py              # PyTorch Dataset + DataLoaders
    ├── model.py                 # ResNet18 transfer-learning classifier
    ├── train.py                  # training loop
    ├── evaluate.py                # precision/recall/F1 + confusion matrix
    ├── predict.py                  # full pipeline: image -> OCR + class
    └── generate_dummy_data.py      # optional: makes fake data to test the pipeline
```

---

## 2. Setup

### a) Python dependencies
```bash
pip install -r requirements.txt
```

### b) Install Tesseract OCR (the actual OCR engine — `pytesseract` is just a Python wrapper around it)

- **Windows:** download the installer from https://github.com/UB-Mannheim/tesseract/wiki, then in `config.py` set:
  ```python
  TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```
- **Mac:** `brew install tesseract`
- **Linux (Debian/Ubuntu):** `sudo apt install tesseract-ocr`

On Mac/Linux you can leave `TESSERACT_CMD = None` in `config.py` as long as `tesseract` is on your PATH (test with `tesseract --version` in a terminal).

---

## 3. What YOU need to change before running this

This is the important part — copy-pasting alone won't work until you do these:

### ★ Change #1 — Your document classes (`config.py`)
```python
CLASSES = ["invoice", "resume", "id_card", "receipt"]
```
Replace this list with **your own document categories**. Whatever you put here must exactly match your folder names in step below (same spelling, same order doesn't matter but names must match).

### ★ Change #2 — Your dataset (`data/raw/`)
Create one sub-folder per class inside `data/raw/`, and drop your images in:
```
data/raw/
    invoice/
        img001.jpg
        img002.jpg
    resume/
        img001.jpg
    id_card/
        ...
    receipt/
        ...
```
Aim for **at least 30–50 images per class** to get a model that isn't just memorizing noise — more is better. If you don't have a real dataset yet and just want to see the pipeline run, use the built-in dummy-data generator first:
```bash
python -m src.generate_dummy_data
```
This fills `data/raw/` with synthetic sample images matching whatever is in `config.CLASSES`, so you can test everything before you have real scans. **Delete this generated data and replace it with your real dataset before doing anything you plan to show off (resume, demo, etc.)** — it's for pipeline testing only.

### ★ Change #3 — Hyperparameters (`config.py`), optional but worth knowing
```python
IMAGE_SIZE = 224        # keep at 224 unless you change the model backbone
BATCH_SIZE = 16          # lower to 8 or 4 if you get "out of memory" errors
NUM_EPOCHS = 10           # raise this if accuracy is still improving at epoch 10
LEARNING_RATE = 1e-4
```

### ★ Change #4 — `DEVICE` (`config.py`)
```python
DEVICE = "cuda"   # auto-falls back to CPU if you don't have a GPU, no action needed
```
Leave as-is — it's safe on machines without a GPU.

---

## 4. Running it

All commands run from the project root (`doc-ocr-classifier/`).

```bash
# (optional) generate fake data to test the pipeline first
python -m src.generate_dummy_data

# train the model — saves the best checkpoint to saved_models/doc_classifier.pt
python -m src.train

# check accuracy / precision / recall / confusion matrix
python -m src.evaluate

# run the full pipeline on one new image (OCR + classification)
python -m src.predict path/to/your/image.jpg

# launch the interactive demo app
streamlit run app.py
```

---

## 5. How each file fits together

| File | Role |
|---|---|
| `src/preprocess.py` | Cleans a raw scanned image so OCR reads it more accurately |
| `src/ocr.py` | Runs Tesseract on the cleaned image, returns text + confidence |
| `src/dataset.py` | Loads `data/raw/<class>/*.jpg` into PyTorch DataLoaders with augmentation |
| `src/model.py` | Defines the CNN — a pretrained ResNet18 with its final layer swapped for your classes |
| `src/train.py` | Trains the model, saves the best checkpoint by validation accuracy |
| `src/evaluate.py` | Loads the saved checkpoint, reports per-class metrics |
| `src/predict.py` | Combines preprocessing + OCR + classification into one call, used by `app.py` |
| `app.py` | Streamlit UI to upload an image and see the whole pipeline run live |

---

## 6. Common issues

- **`No images found under data/raw/...`** — your `config.CLASSES` names don't match your folder names, or the folders are empty.
- **`TesseractNotFoundError`** — Tesseract isn't installed or `TESSERACT_CMD` in `config.py` points to the wrong path.
- **`No trained model found`** (when running `predict.py` or `app.py`) — run `python -m src.train` first.
- **Low accuracy on real data** — you likely need more training images per class (aim for 50+), or more epochs. You can also set `freeze_backbone=False` in `src/train.py`'s call to `build_model()` after a few epochs to fine-tune the whole network, not just the final layer.
