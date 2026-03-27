#!/usr/bin/env python3
"""
Visualize and analyze failure cases from combined_failure_cases.csv.
- Image grid of sample failure cases
- Frequency structure (error types, by source, confidence)
- Accuracy plots (confusion matrix, per-source)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"

# Try to load images
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Allow large images (avoid DecompressionBombError)
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def resolve_image_path(filepath: str, data_root: str = ".") -> str:
    """Convert CSV path to local filesystem path."""
    p = filepath.replace("\\", "/").strip("/")
    # Extract relative path: real/xxx.jpg or fakeV2/fake-v2/xxx.jpg
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
    # Fallback: search by filename in dataset/real and dataset/fakeV2/fake-v2
    fname = os.path.basename(p)
    for sub in ["real", "fakeV2/fake-v2"]:
        candidate = os.path.join(data_root, "dataset", sub, fname)
        if os.path.exists(candidate):
            return candidate
    return path


def load_failure_csv(csv_path: str = "combined_failure_cases.csv") -> pd.DataFrame:
    """Load and prepare failure cases DataFrame."""
    df = pd.read_csv(csv_path)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    # Ensure label/prediction are strings for consistent comparison
    df["label"] = df["label"].astype(str).str.strip()
    df["prediction"] = df["prediction"].astype(str).str.strip()
    df["error_type"] = df.apply(
        lambda r: "Real→Fake" if str(r["label"]) in ("1", "real") else "Fake→Real", axis=1
    )
    return df


def plot_failure_images(
    df: pd.DataFrame,
    data_root: str = ".",
    n_samples: int = 24,
    n_cols: int = 6,
    figsize: tuple = (14, 10),
    save_path: str = "failure_cases_grid.png",
):
    """Display a grid of sample failure case images."""
    if not HAS_PIL:
        print("PIL/Pillow not installed. Skipping image grid.")
        return

    # Sample diverse failure cases (mix of error types)
    real_mask = df["label"].astype(str).isin(("1", "real"))
    fake_mask = df["label"].astype(str).isin(("0", "fake"))
    parts = []
    if real_mask.any():
        n_r = min(n_samples // 2, real_mask.sum())
        parts.append(df[real_mask].sample(n_r))
    if fake_mask.any():
        n_f = min(n_samples // 2, fake_mask.sum())
        parts.append(df[fake_mask].sample(n_f))
    if not parts:
        print("No valid label values (0/1 or real/fake) in CSV. Skipping image grid.")
        return
    sample = pd.concat(parts, ignore_index=True).sample(frac=1).head(n_samples)

    n_show = min(len(sample), n_samples)
    if n_show == 0:
        print("No failure cases to display in grid.")
        return
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, (_, row) in enumerate(sample.iterrows()):
        if idx >= n_show:
            break
        ax = axes[idx]
        local_path = resolve_image_path(row["filepath"], data_root)
        if os.path.exists(local_path):
            img = Image.open(local_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center", fontsize=8, transform=ax.transAxes)
        true_lbl = "Real" if str(row["label"]) in ("1", "real") else "Fake"
        pred_lbl = "Real" if str(row["prediction"]) in ("1", "real") else "Fake"
        conf = row["confidence"] if pd.notna(row["confidence"]) else 0
        ax.set_title(f"True:{true_lbl}→Pred:{pred_lbl}\nconf={conf:.2f}", fontsize=8)
        ax.axis("off")

    for idx in range(n_show, len(axes)):
        axes[idx].axis("off")

    plt.suptitle("Failure Cases: Misclassified Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_frequency_analysis(df: pd.DataFrame, save_dir: str = "."):
    """Plot frequency structure: error types, by source, confidence distribution."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Error type distribution
    ax = axes[0, 0]
    err_counts = df["error_type"].value_counts()
    colors = ["#e74c3c", "#3498db"][:len(err_counts)]
    bars = ax.bar(err_counts.index, err_counts.values, color=colors, edgecolor="black")
    ax.set_title("Error Type Frequency")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(err_counts.values) * 1.15 if len(err_counts) else 1)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 5, int(b.get_height()), ha="center", fontsize=10)

    # 2. By source
    ax = axes[0, 1]
    src_counts = df["source"].value_counts()
    ax.barh(range(len(src_counts)), src_counts.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(src_counts))))
    ax.set_yticks(range(len(src_counts)))
    ax.set_yticklabels(src_counts.index, fontsize=9)
    ax.set_title("Failure Cases by Source")
    ax.set_xlabel("Count")

    # 3. Confidence distribution
    ax = axes[1, 0]
    conf = df["confidence"].dropna()
    if len(conf) > 0:
        ax.hist(conf, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0.5, color="red", linestyle="--", label="0.5")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()

    # 4. Error type by source (stacked)
    ax = axes[1, 1]
    cross = pd.crosstab(df["source"], df["error_type"])
    cross.plot(kind="bar", ax=ax, stacked=False, color=["#e74c3c", "#3498db"])
    ax.set_title("Error Types by Source")
    ax.set_ylabel("Count")
    ax.legend(title="Error Type")
    plt.xticks(rotation=45, ha="right")

    plt.suptitle("Failure Case Frequency Analysis", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "failure_frequency_analysis.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def build_full_confusion_matrix(
    df_failures: pd.DataFrame, data_root: str = "."
):
    """
    Build complete confusion matrix from entire dataset.
    - Scan dataset/real and dataset/fakeV2/fake-v2 for all images
    - Failure cases from CSV have wrong predictions; rest assumed correct
    Returns (cm 2x2 array, total_count, accuracy).
    """
    def to_label(x):
        s = str(x).strip().lower()
        return 1 if s in ("1", "real") else 0

    # Build failure lookup: basename -> (true_label, pred_label)
    failure_lookup = {}
    for _, row in df_failures.iterrows():
        fp = row["filepath"].replace("\\", "/")
        fname = os.path.basename(fp.strip("/"))
        tl = to_label(row["label"])
        pl = to_label(row["prediction"])
        failure_lookup[fname] = (tl, pl)

    # Scan full dataset
    real_dir = os.path.join(data_root, "dataset", "real")
    fake_dir = os.path.join(data_root, "dataset", "fakeV2", "fake-v2")
    cm = np.zeros((2, 2))  # row=true (Real=0,Fake=1), col=pred

    # cm: row 0=Real, row 1=Fake; col 0=Real, col 1=Fake
    for true_label, folder in [(1, real_dir), (0, fake_dir)]:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if f in failure_lookup:
                _, pred = failure_lookup[f]
            else:
                pred = true_label  # correct
            row = 0 if true_label == 1 else 1  # Real=0, Fake=1
            col = 0 if pred == 1 else 1
            cm[row, col] += 1

    total = int(cm.sum())
    correct = int(cm[0, 0] + cm[1, 1])
    acc = correct / total if total > 0 else 0
    return cm, total, acc


def plot_accuracy_analysis(
    df: pd.DataFrame, save_dir: str = ".", data_root: str = "."
):
    """Plot accuracy-related visualizations: full confusion matrix, failure matrix, per-source."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def to_name(x):
        s = str(x).strip().lower()
        if s in ("1", "real"): return "Real"
        if s in ("0", "fake"): return "Fake"
        return "Unknown"

    # 1. FULL confusion matrix (entire dataset)
    ax = axes[0]
    full_cm, total, acc = build_full_confusion_matrix(df, data_root)
    if total > 0:
        im = ax.imshow(full_cm, cmap="Blues", vmin=0, vmax=max(full_cm.max(), 1))
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Real", "Pred: Fake"])
        ax.set_yticklabels(["True: Real", "True: Fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(full_cm[i, j]), ha="center", va="center", fontsize=14)
        ax.set_title(f"Full Confusion Matrix\n(n={total}, acc={acc:.2%})")
        plt.colorbar(im, ax=ax, label="Count")
    else:
        ax.text(0.5, 0.5, "Dataset not found\n(dataset/real, dataset/fakeV2/fake-v2)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Full Confusion Matrix")

    # 2. Failure-only confusion matrix
    ax = axes[1]
    lbl = df["label"].apply(to_name)
    pred = df["prediction"].apply(to_name)
    cm = pd.crosstab(lbl, pred)
    cm = cm.reindex(index=["Real", "Fake"], columns=["Real", "Fake"], fill_value=0)
    cm_arr = np.array(cm.values, dtype=float)
    im = ax.imshow(cm_arr, cmap="Reds", vmin=0, vmax=max(cm_arr.max(), 1))
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Real", "Pred: Fake"])
    ax.set_yticklabels(["True: Real", "True: Fake"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm.values[i, j]), ha="center", va="center", fontsize=14)
    ax.set_title("Failure Cases Only\n(Misclassifications)")
    plt.colorbar(im, ax=ax, label="Count")

    # 3. Per-source failure counts
    ax = axes[2]
    src_counts = df["source"].value_counts()
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(src_counts)))
    ax.barh(range(len(src_counts)), src_counts.values, color=colors)
    ax.set_yticks(range(len(src_counts)))
    ax.set_yticklabels(src_counts.index, fontsize=9)
    ax.set_xlabel("Number of Failure Cases")
    ax.set_title("Failure Count by Source")

    plt.suptitle("Accuracy & Failure Analysis", fontsize=14)
    plt.tight_layout()
    out = os.path.join(save_dir, "failure_accuracy_plots.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    project_dir = Path(__file__).parent
    csv_path = project_dir / "combined_failure_cases.csv"
    data_root = str(project_dir)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}. Run combine_csv.py first.")
        return

    df = load_failure_csv(str(csv_path))
    print(f"Loaded {len(df)} failure cases")
    print(df["error_type"].value_counts())
    print()

    # 1. Image grid
    plot_failure_images(df, data_root=data_root, save_path=str(project_dir / "failure_cases_grid.png"))

    # 2. Frequency analysis
    plot_frequency_analysis(df, save_dir=str(project_dir))

    # 3. Accuracy plots
    plot_accuracy_analysis(df, save_dir=str(project_dir), data_root=data_root)


if __name__ == "__main__":
    main()
