#!/usr/bin/env python3
"""Combine all CSV files in the current folder into one combined CSV file.
Deduplicates rows by matching the last 3 parts of the filepath.
Also creates an intersection file with rows present in ALL source files.
"""

import csv
from pathlib import Path
from typing import Optional


def get_match_key(filepath: str, label: str) -> str:
    """
    Create a match key using the true label and the filename (not including folders).
    """
    import os
    filename = os.path.basename(filepath.replace("\\", "/").strip("/"))
    true_label = str(label).strip().lower()
    return f"{filename}|{true_label}"


def normalize_label(label: str) -> str:
    """Convert label to 1 (real) or 0 (fake)."""
    s = str(label).strip().lower()
    if s in ("1", "real"):
        return "1"
    if s in ("0", "fake"):
        return "0"
    return str(label)


def infer_label_from_path(filepath: str) -> Optional[int]:
    """Infer real (1) or fake (0) from filepath. Returns None if unclear."""
    p = filepath.replace("\\", "/").lower()
    if "/fake" in p or "fakev2" in p:
        return 0
    if "/real" in p:
        return 1
    return None


def detect_and_flip_labels(csv_path: Path) -> bool:
    """Check if file has reversed labels (real=0, fake=1). Returns True if flip needed."""
    match_count = 0
    total = 0
    with open(csv_path, "r", encoding="utf-8") as infile:
        for row in csv.DictReader(infile):
            filepath = row.get("filepath") or row.get("path", "")
            path_label = infer_label_from_path(filepath)
            if path_label is None:
                continue
            csv_label = normalize_label(row.get("true_label", ""))
            if csv_label not in ("0", "1"):
                continue
            total += 1
            if int(csv_label) == path_label:
                match_count += 1
    if total == 0:
        return False
    # If majority don't match, labels are reversed
    return match_count < total / 2


def flip_label(label: str) -> str:
    """Flip 0<->1, Real<->Fake."""
    s = str(label).strip().lower()
    if s in ("1", "real"):
        return "0"
    if s in ("0", "fake"):
        return "1"
    return label


# Find all CSV files in the project folder
project_dir = Path(__file__).parent
csv_files = sorted(project_dir.glob("*.csv"))

# Exclude output files from input list
output_file = project_dir / "combined_failure_cases.csv"
intersection_file = project_dir / "intersection_failure_cases.csv"
csv_files = [f for f in csv_files if f.name not in (output_file.name, intersection_file.name)]
# csv_files = ["failure_cases_with_trainedencoder.csv", "misclassified_test_images.csv", "misclassified_test_images_shannon_entropy.csv"]
if not csv_files:
    print("No CSV files found.")
    exit(1)

print(f"Found {len(csv_files)} CSV files to combine:")
for f in csv_files:
    print(f"  - {f.name}")

# Output: complete filename, key, label, prediction, source, confidence
standard_columns = ["filepath", "key", "label", "prediction", "source", "confidence"]
seen_keys = {}
rows_written = 0

with open(output_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(standard_columns)

    for csv_path in csv_files:
        flip = detect_and_flip_labels(csv_path)
        if flip:
            print(f"  (flipping labels in {csv_path.name})")
        with open(csv_path, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                true_label = flip_label(row.get("true_label", "")) if flip else row.get("true_label", "")
                pred_label = flip_label(row.get("predicted_label", "")) if flip else row.get("predicted_label", "")
                filepath = row.get("filepath") or row.get("path", "")
                match_key = get_match_key(filepath, true_label)
                if match_key in seen_keys:
                    continue
                seen_keys[match_key] = True
                writer.writerow([
                    filepath,
                    match_key,
                    normalize_label(true_label),
                    normalize_label(pred_label),
                    csv_path.name,
                    row.get("confidence", "")
                ])
                rows_written += 1

print(f"\nCombined {rows_written} unique rows (deduplicated by last 3 path parts) into {output_file.name}")

# Build intersection: rows present in ALL files, matched by (filename, true_label)
# Filename = basename only; label normalized to 1 (real) or 0 (fake)
file_sets = []
all_rows = {}  # (filename, label) -> row data from first occurrence

for csv_path in csv_files:
    flip = detect_and_flip_labels(csv_path)
    file_set = set()
    with open(csv_path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            true_label = flip_label(row.get("true_label", "")) if flip else row.get("true_label", "")
            pred_label = flip_label(row.get("predicted_label", "")) if flip else row.get("predicted_label", "")
            filepath = row.get("filepath") or row.get("path", "")
            match_key = get_match_key(filepath, true_label)
            filename = match_key.split("|")[0]  # basename from "filename|label"
            label = normalize_label(true_label)
            file_set.add((filename, label))
            key = (filename, label)
            if key not in all_rows:
                all_rows[key] = {
                    "filepath": filepath,
                    "match_key": match_key,
                    "label": label,
                    "prediction": pred_label,
                    "source": csv_path.name,
                    "confidence": row.get("confidence", "")
                }
    file_sets.append(file_set)

intersection = file_sets[0]
for s in file_sets[1:]:
    intersection = intersection & s

with open(intersection_file, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(standard_columns)
    for (filename, label) in sorted(intersection):
        r = all_rows[(filename, label)]
        writer.writerow([r["filepath"], r["match_key"], r["label"], normalize_label(r["prediction"]), r["source"], r["confidence"]])

print(f"Intersection: {len(intersection)} rows (in all {len(csv_files)} files) into {intersection_file.name}")
