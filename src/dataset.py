"""
src/dataset.py

Loads document images from data/raw/<class_name>/*.jpg into a PyTorch
Dataset, and builds train/validation DataLoaders.

Expected folder layout (folder names must match config.CLASSES exactly):

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
"""

import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import config


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class DocumentDataset(Dataset):
    def __init__(self, root_dir: str, classes: list[str], transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        self.transform = transform
        self.samples = self._build_index()

    def _build_index(self):
        samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[WARNING] Expected class folder not found: {cls_dir}")
                continue

            for ext in IMG_EXTENSIONS:
                for path in glob.glob(os.path.join(cls_dir, f"*{ext}")):
                    samples.append((path, self.class_to_idx[cls_name]))

        if len(samples) == 0:
            raise RuntimeError(
                f"No images found under {self.root_dir}. "
                f"Check that config.CLASSES matches your folder names, "
                f"and that images have one of these extensions: {IMG_EXTENSIONS}"
            )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool = True):
    """
    ImageNet normalization stats are used because the CNN backbone (ResNet18)
    is pretrained on ImageNet -- see src/model.py.
    """
    base = [
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    if train:
        # Light augmentation helps a lot when you only have a small labeled dataset
        augment = [
            transforms.RandomRotation(5),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(augment + base)

    return transforms.Compose(base)


def get_dataloaders():
    """
    Builds train + validation DataLoaders from config.RAW_DATA_DIR.

    NOTE: we build two separate DocumentDataset instances (one with train-time
    augmentation, one without) rather than mutating a shared dataset's
    .transform after random_split -- that's a common bug, since random_split's
    Subset objects both point at the *same* underlying dataset object.
    """
    train_dataset_full = DocumentDataset(
        root_dir=config.RAW_DATA_DIR,
        classes=config.CLASSES,
        transform=get_transforms(train=True),
    )
    val_dataset_full = DocumentDataset(
        root_dir=config.RAW_DATA_DIR,
        classes=config.CLASSES,
        transform=get_transforms(train=False),
    )

    val_size = int(len(train_dataset_full) * config.VAL_SPLIT)
    train_size = len(train_dataset_full) - val_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_indices, val_indices = random_split(
        range(len(train_dataset_full)), [train_size, val_size], generator=generator
    )

    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader
