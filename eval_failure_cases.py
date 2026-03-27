#!/usr/bin/env python3
"""
Evaluate models on failure cases only (from combined_failure_cases.csv).
Resolves paths to local dataset, filters to existing files, runs ViT, CLIP, ensemble.
Usage: python eval_failure_cases.py [--data-dir .] [--csv combined_failure_cases.csv]
"""
import argparse
import os
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import timm
import open_clip
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    roc_auc_score,
)

Image.MAX_IMAGE_PIXELS = None


def resolve_image_path(filepath: str, data_root: str = ".") -> str:
    """Convert CSV path to local filesystem path."""
    p = filepath.replace("\\", "/").strip("/")
    for prefix in [
        "/kaggle/input/dalle-recognition-dataset/",
        "/root/.cache/kagglehub/datasets/superpotato9/dalle-recognition-dataset/versions/7/",
    ]:
        if prefix in p:
            p = p.split(prefix, 1)[-1]
            break
    if p.startswith("data/"):
        path = os.path.join(data_root, p)
    else:
        path = os.path.join(data_root, "dataset", p)
    if os.path.exists(path):
        return path
    fname = os.path.basename(p)
    for sub in ["real", "fakeV2/fake-v2"]:
        candidate = os.path.join(data_root, "dataset", sub, fname)
        if os.path.exists(candidate):
            return candidate
    return path


class FailureCaseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.loc[idx]
        img_path = row["local_path"]
        label = int(row["label"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long), img_path


CONFIDENCE_BUCKETS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
MAX_EXAMPLES_PER_BUCKET = 8
N_COLS_GRID = 4


def save_examples_by_bucket(
    indices, confidences, labels, preds, paths, model_name, out_dir, n_cols=N_COLS_GRID
):
    """Save example images for each confidence bucket. indices/confidences/labels/preds/paths aligned."""
    os.makedirs(out_dir, exist_ok=True)
    for lo, hi in CONFIDENCE_BUCKETS:
        mask = (confidences >= lo) & (confidences < hi)
        bucket_indices = np.where(mask)[0]
        if len(bucket_indices) == 0:
            continue
        n_show = min(MAX_EXAMPLES_PER_BUCKET, len(bucket_indices))
        sel = bucket_indices[:n_show]

        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        for idx, i in enumerate(sel):
            ax = axes.flatten()[idx]
            path = paths[i]
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                ax.imshow(img)
            else:
                ax.set_facecolor("#f0f0f0")
                ax.text(0.5, 0.5, "Not found", ha="center", va="center", transform=ax.transAxes)
            true_lbl = "Real" if labels[i] == 1 else "Fake"
            pred_lbl = "Real" if preds[i] == 1 else "Fake"
            ax.set_title(f"T:{true_lbl}→P:{pred_lbl} conf={confidences[i]:.3f}", fontsize=9)
            ax.axis("off")
        for idx in range(n_show, axes.size):
            axes.flatten()[idx].axis("off")
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        fname = f"{safe_name}_conf_{lo:.1f}-{hi:.1f}_n{len(bucket_indices)}.png"
        fig.suptitle(f"{model_name} | conf [{lo:.1f},{hi:.1f}) | {len(bucket_indices)} total", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close()


def get_failure_loader(csv_path, data_root, batch_size=32):
    """Load failure cases from CSV, resolve to local paths, return DataLoader."""
    df = pd.read_csv(csv_path)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["prediction"] = df["prediction"].fillna(0).astype(int)

    local_paths = []
    for fp in df["filepath"]:
        lp = resolve_image_path(fp, data_root)
        local_paths.append(lp)
    df["local_path"] = local_paths

    # Keep only rows where file exists and is an image
    def is_image(path):
        return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    df = df[
        df["local_path"].apply(lambda p: os.path.exists(p) and is_image(p))
    ].reset_index(drop=True)
    if len(df) == 0:
        raise FileNotFoundError(
            "No failure case images found in dataset. Check data_root and dataset/real, dataset/fakeV2/fake-v2."
        )

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = FailureCaseDataset(df, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, df


def print_metrics(name, labels, preds, probs=None):
    acc = accuracy_score(labels, preds)
    print(f"\n{name}")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(labels, preds, labels=[0, 1])
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

# Synonymous prompts for encoder+CLIP (different from training prompts to test generalization)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--csv", default="combined_failure_cases.csv")
    parser.add_argument("--vit-weights", default="best_vit (1).pth")
    parser.add_argument("--proj-weights", default="projection_head.pth")
    parser.add_argument("--method", choices=["vote", "average"], default="average")
    parser.add_argument("--save-histograms", default="failure_cases_confidence_histograms.png")
    parser.add_argument("--examples-dir", default="failure_case_examples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading failure cases from {args.csv}...")
    loader, df = get_failure_loader(args.csv, args.data_dir)
    print(f"Evaluating on {len(df)} failure cases (files found in dataset)")

    # 1. Save examples from CSV failure cases (original model's confidence)
    csv_conf = df["confidence"].fillna(0.5).values
    csv_labels = df["label"].values
    csv_preds = df["prediction"].values
    paths = df["local_path"].values
    csv_dir = os.path.join(args.examples_dir, "01_csv_failures")
    save_examples_by_bucket(
        np.arange(len(df)), csv_conf, csv_labels, csv_preds, paths,
        "CSV failures (original)", csv_dir
    )
    print(f"Saved CSV failure examples to {csv_dir}")

    # ViT classifier
    vit = timm.create_model("vit_base_patch32_224", pretrained=False, num_classes=2)
    vit.load_state_dict(torch.load(args.vit_weights, map_location="cpu"))
    vit.to(device).eval()

    all_logits_vit = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="ViT"):
            out = vit(images.to(device))
            all_logits_vit.append(out.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_logits_vit = np.concatenate(all_logits_vit, axis=0)
    all_labels = np.array(all_labels)
    preds_vit = all_logits_vit.argmax(1)
    logits_stable = all_logits_vit - all_logits_vit.max(1, keepdims=True)
    probs_vit = np.exp(logits_stable) / np.exp(logits_stable).sum(1, keepdims=True)

    # Collect for histograms: (name, confidence array)
    hist_data = []
    hist_data.append(("ViT classifier", np.max(probs_vit, axis=1)))
    conf_vit = np.max(probs_vit, axis=1)
    mis_vit = preds_vit != all_labels
    if mis_vit.any():
        m_conf, m_lbl, m_pred, m_paths = conf_vit[mis_vit], all_labels[mis_vit], preds_vit[mis_vit], paths[mis_vit]
        save_examples_by_bucket(
            np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
            "ViT classifier", os.path.join(args.examples_dir, "02_ViT_misclassified")
        )

    print_metrics("ViT classifier (failure cases only)", all_labels, preds_vit, probs_vit)

    # Multi-prompt CLIP
    model_clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model_clip.to(device).eval()
    with torch.no_grad():
        fake_feats = model_clip.encode_text(tokenizer(PROMPTS_FAKE).to(device))
        real_feats = model_clip.encode_text(tokenizer(PROMPTS_REAL).to(device))
        text_fake = F.normalize(fake_feats.mean(0, keepdim=True), dim=-1)
        text_real = F.normalize(real_feats.mean(0, keepdim=True), dim=-1)
        text_features = torch.cat([text_fake, text_real], dim=0).to(device)

    all_preds_clip = []
    all_probs_clip = []
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="CLIP"):
            img_feats = model_clip.encode_image(images.to(device))
            img_feats = F.normalize(img_feats, dim=-1)
            logits = img_feats @ text_features.T
            probs = F.softmax(logits, dim=1)
            all_preds_clip.extend(logits.argmax(1).cpu().numpy())
            all_probs_clip.extend(probs.cpu().numpy())

    all_preds_clip = np.array(all_preds_clip)
    all_probs_clip = np.array(all_probs_clip)
    conf_clip = np.max(all_probs_clip, axis=1)
    hist_data.append(("Multi-prompt CLIP", conf_clip))
    mis_clip = all_preds_clip != all_labels
    if mis_clip.any():
        m_conf, m_lbl, m_pred, m_paths = conf_clip[mis_clip], all_labels[mis_clip], all_preds_clip[mis_clip], paths[mis_clip]
        save_examples_by_bucket(np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
            "Multi-prompt CLIP", os.path.join(args.examples_dir, "03_CLIP_misclassified"))
    print_metrics("Multi-prompt CLIP (failure cases only)", all_labels, all_preds_clip, all_probs_clip)

    # Ensemble (ViT + encoder+CLIP) if projection head exists
    if os.path.exists(args.proj_weights):
        encoder = TrainedViTEncoder(args.vit_weights).to(device).eval()
        proj = ProjectionHead(768, 512).to(device)
        proj.load_state_dict(torch.load(args.proj_weights, map_location="cpu"))
        proj.eval()

        # Original training prompts
        text_feats_orig = F.normalize(
            model_clip.encode_text(tokenizer(["AI generated artwork", "Real artwork"]).to(device)).float(),
            dim=-1
        ).to(device)

        all_logits_enc = []
        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc="Encoder+CLIP (original)"):
                feats = encoder(images.to(device))
                proj_feats = F.normalize(proj(feats), dim=-1)
                logits = (proj_feats @ text_feats_orig.T).float().cpu().numpy()
                all_logits_enc.append(logits)

        all_logits_enc = np.concatenate(all_logits_enc, axis=0)
        preds_enc = all_logits_enc.argmax(1)
        enc_scale = 10.0 / (np.max(np.abs(all_logits_enc)) + 1e-8)
        enc_scaled = all_logits_enc * enc_scale
        enc_stable = enc_scaled - enc_scaled.max(1, keepdims=True)
        probs_enc = np.exp(enc_stable) / np.exp(enc_stable).sum(1, keepdims=True)
        conf_enc = np.max(probs_enc, axis=1)
        hist_data.append(("Encoder+CLIP (original)", conf_enc))
        mis_enc = preds_enc != all_labels
        if mis_enc.any():
            m_conf, m_lbl, m_pred, m_paths = conf_enc[mis_enc], all_labels[mis_enc], preds_enc[mis_enc], paths[mis_enc]
            save_examples_by_bucket(np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
                "Encoder+CLIP (original)", os.path.join(args.examples_dir, "04_EncoderCLIP_orig_misclassified"))

        print_metrics("Encoder + projection + CLIP, original prompts (failure cases only)", all_labels, preds_enc, probs_enc)

        # Synonymous prompts (test generalization)
        with torch.no_grad():
            syn_fake = model_clip.encode_text(tokenizer(PROMPTS_ENC_SYNONYMOUS_FAKE).to(device))
            syn_real = model_clip.encode_text(tokenizer(PROMPTS_ENC_SYNONYMOUS_REAL).to(device))
            text_feats_syn = F.normalize(
                torch.cat([
                    syn_fake.mean(0, keepdim=True),
                    syn_real.mean(0, keepdim=True),
                ], dim=0).float(),
                dim=-1
            ).to(device)

        all_logits_enc_syn = []
        with torch.no_grad():
            for images, labels, _ in tqdm(loader, desc="Encoder+CLIP (synonymous)"):
                feats = encoder(images.to(device))
                proj_feats = F.normalize(proj(feats), dim=-1)
                logits = (proj_feats @ text_feats_syn.T).float().cpu().numpy()
                all_logits_enc_syn.append(logits)

        all_logits_enc_syn = np.concatenate(all_logits_enc_syn, axis=0)
        preds_enc_syn = all_logits_enc_syn.argmax(1)
        enc_syn_scale = 10.0 / (np.max(np.abs(all_logits_enc_syn)) + 1e-8)
        enc_syn_scaled = all_logits_enc_syn * enc_syn_scale
        enc_syn_stable = enc_syn_scaled - enc_syn_scaled.max(1, keepdims=True)
        probs_enc_syn = np.exp(enc_syn_stable) / np.exp(enc_syn_stable).sum(1, keepdims=True)
        conf_enc_syn = np.max(probs_enc_syn, axis=1)
        hist_data.append(("Encoder+CLIP (synonymous)", conf_enc_syn))
        mis_enc_syn = preds_enc_syn != all_labels
        if mis_enc_syn.any():
            m_conf, m_lbl, m_pred, m_paths = conf_enc_syn[mis_enc_syn], all_labels[mis_enc_syn], preds_enc_syn[mis_enc_syn], paths[mis_enc_syn]
            save_examples_by_bucket(np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
                "Encoder+CLIP (synonymous)", os.path.join(args.examples_dir, "05_EncoderCLIP_syn_misclassified"))

        print_metrics("Encoder + projection + CLIP, synonymous prompts (failure cases only)", all_labels, preds_enc_syn, probs_enc_syn)

        if args.method == "average":
            avg_logits = 0.5 * (all_logits_vit + all_logits_enc)
            ensemble_preds = avg_logits.argmax(1)
            avg_stable = avg_logits - avg_logits.max(1, keepdims=True)
            ensemble_probs = np.exp(avg_stable) / np.exp(avg_stable).sum(1, keepdims=True)
            conf_ens = np.max(ensemble_probs, axis=1)
            hist_data.append((f"Ensemble ({args.method})", conf_ens))
            mis_ens = ensemble_preds != all_labels
            if mis_ens.any():
                m_conf, m_lbl, m_pred, m_paths = conf_ens[mis_ens], all_labels[mis_ens], ensemble_preds[mis_ens], paths[mis_ens]
                save_examples_by_bucket(np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
                    f"Ensemble ({args.method})", os.path.join(args.examples_dir, "06_Ensemble_misclassified"))
            print_metrics("Ensemble average logits (failure cases only)", all_labels, ensemble_preds, ensemble_probs)
        else:
            ensemble_preds = (preds_vit + preds_enc >= 1).astype(int)
            ensemble_probs = 0.5 * (probs_vit + probs_enc)
            conf_ens = np.max(ensemble_probs, axis=1)
            hist_data.append((f"Ensemble ({args.method})", conf_ens))
            mis_ens = ensemble_preds != all_labels
            if mis_ens.any():
                m_conf, m_lbl, m_pred, m_paths = conf_ens[mis_ens], all_labels[mis_ens], ensemble_preds[mis_ens], paths[mis_ens]
                save_examples_by_bucket(np.arange(len(m_conf)), m_conf, m_lbl, m_pred, m_paths,
                    f"Ensemble ({args.method})", os.path.join(args.examples_dir, "06_Ensemble_misclassified"))
            print_metrics("Ensemble majority vote (failure cases only)", all_labels, ensemble_preds, ensemble_probs)
    else:
        print("\n(Optional: run train_projection_head.py to add encoder+CLIP and ensemble metrics)")

    # Plot confidence histograms
    n_models = len(hist_data)
    n_cols = 2
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, (name, conf) in enumerate(hist_data):
        ax = axes.flatten()[idx]
        ax.hist(conf, bins=30, range=(0, 1), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Confidence (max prob)")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.axvline(conf.mean(), color="red", linestyle="--", label=f"Mean={conf.mean():.3f}")
        ax.legend()
    for idx in range(n_models, axes.size):
        axes.flatten()[idx].set_visible(False)
    fig.suptitle("Confidence histograms on failure cases", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(args.save_histograms, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confidence histograms to {args.save_histograms}")


if __name__ == "__main__":
    main()
