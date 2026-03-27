#!/usr/bin/env python3
"""
Multi-prompt CLIP evaluation: try multiple text prompts and average scores.
Usage: python eval_clip_multi_prompt.py [--data-dir .]
"""
import argparse
import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)

from model_utils import get_loaders


# Multiple prompts per class; we average the text features
PROMPTS_FAKE = [
    "AI generated artwork",
    "AI-generated image",
    "synthetic digital art",
    "Dall-E generated",
    "computer generated art",
]
PROMPTS_REAL = [
    "Real artwork",
    "human-made painting",
    "photograph of real art",
    "traditional artwork",
    "hand-painted art",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _, _, _ = get_loaders(args.data_dir)

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()

    # Encode all prompts and average per class
    with torch.no_grad():
        fake_tokens = tokenizer(PROMPTS_FAKE).to(device)
        real_tokens = tokenizer(PROMPTS_REAL).to(device)
        fake_feats = model.encode_text(fake_tokens)
        real_feats = model.encode_text(real_tokens)
        text_fake = F.normalize(fake_feats.mean(0, keepdim=True), dim=-1)
        text_real = F.normalize(real_feats.mean(0, keepdim=True), dim=-1)
        text_features = torch.cat([text_fake, text_real], dim=0)  # (2, 512)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            img_feats = model.encode_image(images)
            img_feats = F.normalize(img_feats, dim=-1)
            logits = img_feats @ text_features.T
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nMulti-prompt CLIP (avg of {len(PROMPTS_FAKE)} prompts per class)")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")

    # Confusion matrix (rows=true, cols=pred) [Fake=0, Real=1]
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    print("              Pred:Fake  Pred:Real")
    print(f"True: Fake    {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"True: Real    {cm[1,0]:8d}  {cm[1,1]:8d}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0
    )
    print("\nPer-class metrics:")
    print(f"  Fake (0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}, Support={support[0]}")
    print(f"  Real (1): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}, Support={support[1]}")

    # Macro / weighted averages
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Fake", "Real"], digits=4))

    # ROC-AUC (binary: use prob of Real class)
    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC: N/A (need both classes present)")


if __name__ == "__main__":
    main()
