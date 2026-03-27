#!/usr/bin/env python3
"""
Ensemble evaluation: combine ViT classifier and trained encoder + projection head.
Voting or average logits. Requires: best_vit.pth, projection_head.pth (optional).
Usage: python eval_ensemble.py [--vit-weights best_vit.pth] [--proj-weights projection_head.pth]
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
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

# Synonymous prompts for encoder+CLIP (test generalization beyond training prompts)
PROMPTS_ENC_SYNONYMOUS_FAKE = [
    "machine-generated art",
    "synthetic image",
    "algorithmically created artwork",
    "AI-created image",
    "computer-synthesized art",
]
PROMPTS_ENC_SYNONYMOUS_REAL = [
    "genuine artwork",
    "human-painted artwork",
    "original artist creation",
    "manually created art",
    "authentic painting",
]


def print_metrics(name, labels, preds, probs=None):
    """Print accuracy, confusion matrix, per-class metrics, classification report, ROC-AUC."""
    acc = accuracy_score(labels, preds)
    print(f"\n{name}")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix (rows=True, cols=Predicted):")
    print("              Pred:Fake  Pred:Real")
    print(f"True: Fake    {cm[0,0]:8d}  {cm[0,1]:8d}")
    print(f"True: Real    {cm[1,0]:8d}  {cm[1,1]:8d}")

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    print("\nPer-class metrics:")
    print(f"  Fake (0): Precision={precision[0]:.4f}, Recall={recall[0]:.4f}, F1={f1[0]:.4f}, Support={support[0]}")
    print(f"  Real (1): Precision={precision[1]:.4f}, Recall={recall[1]:.4f}, F1={f1[1]:.4f}, Support={support[1]}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Fake", "Real"], digits=4))

    if probs is not None:
        try:
            auc = roc_auc_score(labels, probs[:, 1])
            print(f"ROC-AUC: {auc:.4f}")
        except ValueError:
            print("ROC-AUC: N/A (need both classes present)")


class ProjectionHead(nn.Module):
    """Must match architecture in train_projection_head.py."""

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
    def __init__(self, checkpoint_path, model_name="vit_base_patch32_224", num_classes=2):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        self.model.eval()

    def forward(self, x):
        return self.model.forward_features(x)[:, 0, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vit-weights", default="best_vit.pth")
    parser.add_argument("--proj-weights", default="projection_head.pth", help="Optional; if provided, use encoder+CLIP")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--method", choices=["vote", "average"], default="vote")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, _, _, _ = get_loaders(args.data_dir)

    # Model 1: ViT classifier
    vit = timm.create_model("vit_base_patch32_224", pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load(args.vit_weights, map_location="cpu"))
    vit.to(device).eval()

    use_encoder = False
    if args.proj_weights and __import__("os").path.exists(args.proj_weights):
        use_encoder = True
        encoder = TrainedViTEncoder(args.vit_weights).to(device).eval()
        proj = ProjectionHead(768, 512).to(device)
        proj.load_state_dict(torch.load(args.proj_weights, map_location="cpu"))
        proj.eval()
        model_clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        with torch.no_grad():
            text_feats = F.normalize(
                model_clip.encode_text(tokenizer(["AI generated artwork", "Real artwork"]).to(device)).float(),
                dim=-1
            ).to(device)

    all_logits_vit = []
    all_logits_enc = [] if use_encoder else None
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            out_vit = vit(images)
            all_logits_vit.append(out_vit.cpu().numpy())
            all_labels.extend(labels.numpy())

            if use_encoder:
                feats = encoder(images)
                proj_feats = F.normalize(proj(feats), dim=-1)
                logits_enc = (proj_feats @ text_feats.T).float().cpu().numpy()
                all_logits_enc.append(logits_enc)

    all_logits_vit = np.concatenate(all_logits_vit, axis=0)
    all_labels = np.array(all_labels)

    preds_vit = all_logits_vit.argmax(1)
    logits_stable = all_logits_vit - all_logits_vit.max(1, keepdims=True)
    probs_vit = np.exp(logits_stable) / np.exp(logits_stable).sum(1, keepdims=True)
    print_metrics("ViT classifier alone", all_labels, preds_vit, probs_vit)

    if use_encoder:
        all_logits_enc = np.concatenate(all_logits_enc, axis=0)
        preds_enc = all_logits_enc.argmax(1)
        enc_scale = 10.0 / (np.max(np.abs(all_logits_enc)) + 1e-8)
        enc_scaled = all_logits_enc * enc_scale
        enc_stable = enc_scaled - enc_scaled.max(1, keepdims=True)
        probs_enc = np.exp(enc_stable) / np.exp(enc_stable).sum(1, keepdims=True)
        print_metrics("Encoder + projection + CLIP (original prompts)", all_labels, preds_enc, probs_enc)

        # Synonymous prompts
        with torch.no_grad():
            syn_fake = model_clip.encode_text(tokenizer(PROMPTS_ENC_SYNONYMOUS_FAKE).to(device))
            syn_real = model_clip.encode_text(tokenizer(PROMPTS_ENC_SYNONYMOUS_REAL).to(device))
            text_feats_syn = F.normalize(
                torch.cat([syn_fake.mean(0, keepdim=True), syn_real.mean(0, keepdim=True)], dim=0).float(),
                dim=-1
            ).to(device)

        all_logits_enc_syn = []
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc="Encoder+CLIP (synonymous)"):
                feats = encoder(images)
                proj_feats = F.normalize(proj(feats), dim=-1)
                all_logits_enc_syn.append((proj_feats @ text_feats_syn.T).float().cpu().numpy())

        all_logits_enc_syn = np.concatenate(all_logits_enc_syn, axis=0)
        preds_enc_syn = all_logits_enc_syn.argmax(1)
        enc_syn_scale = 10.0 / (np.max(np.abs(all_logits_enc_syn)) + 1e-8)
        enc_syn_scaled = all_logits_enc_syn * enc_syn_scale
        enc_syn_stable = enc_syn_scaled - enc_syn_scaled.max(1, keepdims=True)
        probs_enc_syn = np.exp(enc_syn_stable) / np.exp(enc_syn_stable).sum(1, keepdims=True)
        print_metrics("Encoder + projection + CLIP (synonymous prompts)", all_labels, preds_enc_syn, probs_enc_syn)

        if args.method == "vote":
            ensemble_preds = (preds_vit + preds_enc >= 1).astype(int)
            ensemble_probs = 0.5 * (probs_vit + probs_enc)
            print_metrics("Ensemble (majority vote)", all_labels, ensemble_preds, ensemble_probs)
        else:
            avg_logits = 0.5 * (all_logits_vit + all_logits_enc)
            ensemble_preds = avg_logits.argmax(1)
            avg_stable = avg_logits - avg_logits.max(1, keepdims=True)
            ensemble_probs = np.exp(avg_stable) / np.exp(avg_stable).sum(1, keepdims=True)
            print_metrics("Ensemble (average logits)", all_labels, ensemble_preds, ensemble_probs)
    else:
        print("\nRun train_projection_head.py first to enable encoder ensemble.")


if __name__ == "__main__":
    main()
