#!/usr/bin/env python3
"""
Train the projection head (768→512) to map trained ViT features to CLIP text space.
Uses contrastive-style loss: align image features with correct text features.
Usage: python train_projection_head.py --vit-weights best_vit.pth [--epochs 10]
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
from tqdm import tqdm

from model_utils import get_loaders


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, output_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim, bias=False),
        )

    def forward(self, x):
        return self.proj(x)


class TrainedViTEncoder(nn.Module):
    """Extract 768-dim features from trained ViT (before classification head)."""

    def __init__(self, checkpoint_path, model_name="vit_base_patch32_224", num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        self.model.eval()

    def forward(self, x):
        feats = self.model.forward_features(x)  # (B, N, 768)
        return feats[:, 0, :]  # CLS token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-weights", default="best_vit.pth", help="Path to trained ViT weights")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="projection_head.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, _, _, _ = get_loaders(args.data_dir)

    # Load CLIP text features (frozen)
    model_clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = ["AI generated artwork", "Real artwork"]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model_clip.encode_text(text_tokens)
        text_features = F.normalize(text_features.float(), dim=-1)
    text_features = text_features.to(device)

    # ViT encoder (frozen) + trainable projection head
    vit_encoder = TrainedViTEncoder(args.vit_weights).to(device)
    for p in vit_encoder.parameters():
        p.requires_grad = False
    proj_head = ProjectionHead(768, 512).to(device)
    optimizer = torch.optim.Adam(proj_head.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        proj_head.train()
        correct, total = 0, 0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                img_feats = vit_encoder(images)
            proj_feats = proj_head(img_feats)
            proj_feats = F.normalize(proj_feats, dim=-1)
            logits = proj_feats @ text_features.T
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        proj_head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                img_feats = vit_encoder(images)
                proj_feats = F.normalize(proj_head(img_feats), dim=-1)
                preds = (proj_feats @ text_features.T).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    torch.save(proj_head.state_dict(), args.output)
    print(f"Saved projection head to {args.output}")


if __name__ == "__main__":
    main()
