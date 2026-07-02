"""
src/generate_dummy_data.py

OPTIONAL helper -- NOT part of the core project.

Generates a small synthetic dataset of fake "documents" (plain images with
class-specific text and shapes drawn on them) so you can run the full
pipeline end-to-end immediately, before you have your own scanned dataset.

Run with:  python -m src.generate_dummy_data
Then:      python -m src.train

Once you have real scanned documents, delete data/raw/* and replace it with
your own images, organized the same way (one sub-folder per class).
"""

import os
import random

from PIL import Image, ImageDraw, ImageFont

import config

SAMPLES_PER_CLASS = 40
IMG_SIZE = (400, 500)


def make_fake_document(text_lines, out_path, seed):
    random.seed(seed)
    img = Image.new("RGB", IMG_SIZE, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # a few random lines/boxes to simulate document structure
    for _ in range(random.randint(2, 5)):
        x0, y0 = random.randint(10, 200), random.randint(10, 400)
        x1, y1 = x0 + random.randint(50, 150), y0 + random.randint(10, 30)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0))

    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 25

    # light synthetic noise so preprocessing has something to clean up
    noisy = img.copy()
    px = noisy.load()
    for _ in range(500):
        x = random.randint(0, IMG_SIZE[0] - 1)
        y2 = random.randint(0, IMG_SIZE[1] - 1)
        px[x, y2] = (random.randint(200, 255),) * 3

    noisy.save(out_path)


CLASS_TEXT_TEMPLATES = {
    "invoice": ["INVOICE #{n}", "Bill To: Customer {n}", "Total Due: ${amt}"],
    "resume": ["Name: Candidate {n}", "Experience: {n} years", "Skills: Python, SQL"],
    "id_card": ["ID CARD", "ID No: {n}", "Valid Thru: 20{n}"],
    "receipt": ["RECEIPT", "Store #{n}", "Amount Paid: ${amt}"],
}


def main():
    for cls_name in config.CLASSES:
        cls_dir = os.path.join(config.RAW_DATA_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        template = CLASS_TEXT_TEMPLATES.get(cls_name, [f"{cls_name.upper()} DOCUMENT", "Sample text {n}"])

        for i in range(SAMPLES_PER_CLASS):
            lines = [t.format(n=i, amt=round(random.uniform(10, 500), 2)) for t in template]
            out_path = os.path.join(cls_dir, f"{cls_name}_{i:03d}.png")
            make_fake_document(lines, out_path, seed=i)

        print(f"Generated {SAMPLES_PER_CLASS} dummy images for class '{cls_name}' -> {cls_dir}")

    print("\nDone. You can now run: python -m src.train")


if __name__ == "__main__":
    main()
