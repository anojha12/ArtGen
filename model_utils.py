"""
Shared utilities for ArtGen model scripts: dataset, transforms, data loaders.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# Allow large images (PIL default ~178M pixels can trigger DecompressionBombError)
Image.MAX_IMAGE_PIXELS = None

from torchvision import transforms
from sklearn.model_selection import train_test_split


def create_dataframe(data_dir="."):
    """Create DataFrame with image paths and labels. Real=1, Fake=0."""
    fake_dir = os.path.join(data_dir, "dataset/fakeV2/fake-v2")
    real_dir = os.path.join(data_dir, "dataset/real")
    fake_images = [
        os.path.join(fake_dir, f)
        for f in os.listdir(fake_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    real_images = [
        os.path.join(real_dir, f)
        for f in os.listdir(real_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    data = pd.DataFrame({
        "filename": fake_images + real_images,
        "label": [0] * len(fake_images) + [1] * len(real_images),
    })
    return data


class ArtDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "filename"]
        label = self.dataframe.loc[idx, "label"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), img_path


def get_transforms():
    """Standard ImageNet normalization transforms."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def get_loaders(data_dir=".", batch_size=32, test_size=0.2, val_size=0.25, seed=42):
    """Create train/val/test loaders with stratified split."""
    data = create_dataframe(data_dir)
    train_df, test_df = train_test_split(
        data, test_size=test_size, stratify=data["label"], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"], random_state=seed
    )
    train_tf, val_tf = get_transforms()
    train_loader = torch.utils.data.DataLoader(
        ArtDataset(train_df, transform=train_tf),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        ArtDataset(val_df, transform=val_tf),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        ArtDataset(test_df, transform=val_tf),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def get_class_weights(data_dir="."):
    """Compute class weights for imbalanced data (inverse frequency)."""
    data = create_dataframe(data_dir)
    counts = data["label"].value_counts().sort_index()
    total = len(data)
    weights = torch.tensor([total / (2 * c) for c in counts], dtype=torch.float32)
    return weights
