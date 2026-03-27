#!/usr/bin/env python3
"""
Train ViT with class weights to handle imbalanced data (AI vs Real).
Usage: python train_vit_class_weights.py [--epochs 4] [--data-dir .]
"""
import argparse
import torch
import torch.nn as nn
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

from model_utils import get_loaders, get_class_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output", default="best_vit (1).pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    class_weights = get_class_weights(args.data_dir).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    model = timm.create_model("vit_base_patch32_224", pretrained=True, num_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0
    for epoch in range(args.epochs):
        model.train()
        correct, total = 0, 0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)
            print(f"  Saved best to {args.output}")

    model.load_state_dict(torch.load(args.output))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            preds = model(images.to(device)).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
