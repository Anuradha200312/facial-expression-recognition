#!/usr/bin/env python3
"""
YOLO Stratified Train/Val/Test Splitter
========================================

Splits a YOLO-format detection dataset (images/ + labels/) into
train/val/test folders while trying to preserve class ratios across the
splits, with explicit handling for rare/imbalanced classes. Also writes a
Roboflow-style data.yaml for YOLO training (YOLOv5/v8/v11 compatible).

-------------------------------------------------------------------------
WHY "STRATIFIED" IS DIFFERENT FOR OBJECT DETECTION vs. CLASSIFICATION
-------------------------------------------------------------------------
In image classification, each image has exactly ONE label, so you can
stratify directly on that label (like the sklearn train_test_split(stratify=...)
pattern). In object detection, a single image can contain boxes of SEVERAL
different classes at once, so there's no single "the" label to stratify on.

This script handles that by assigning each image a "dominant class" = the
class with the most box instances in that image (ties broken by lowest
class id). Images are then split so that each dominant-class group is
divided into train/val/test using the same ratios, which keeps the overall
class distribution roughly consistent across splits even though object
detection labels are multi-class per image.

-------------------------------------------------------------------------
CLASS-IMBALANCE HANDLING
-------------------------------------------------------------------------
Rather than relying on sklearn's stratify (which raises an error if any
class has fewer members than the number of splits), this script uses a
manual, imbalance-aware allocator per dominant-class group:
    - >= 3 images in the group -> split proportionally across train/val/test,
      guaranteeing at least 1 image in val and 1 in test if their ratio > 0
      (so rare classes still show up for evaluation, not just training).
    - 2 images in the group -> 1 to train, 1 to val (skipped from test,
      logged as a warning).
    - 1 image in the group -> goes to train only (logged as a warning,
      since there's nothing left to validate/test that class with).
This guarantees the split never crashes on imbalanced datasets and gives
you a clear log of exactly which classes are too rare to fully evaluate.

-------------------------------------------------------------------------
HARD-CODED SETTINGS (edit these constants below to configure the script)
-------------------------------------------------------------------------
    SOURCE_DIR   - path to the source dataset (must contain images/ + labels/)
    OUTPUT_DIR   - where the split dataset + data.yaml will be written
    TRAIN_RATIO / VAL_RATIO / TEST_RATIO  - must sum to 1.0
    CLASS_NAMES  - class names in class-id order, for data.yaml (used if no
                   classes.txt is found in SOURCE_DIR); set to None to
                   auto-detect from classes.txt or fall back to "class_0",
                   "class_1", ...
    SEED         - random seed for reproducibility

Output layout (Roboflow / YOLOv5-v11 style):

    OUTPUT_DIR/
        train/
            images/
            labels/
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml

Requirements:
    pip install pyyaml
    (Pillow is NOT required by this script - it only moves/copies files.)

Run:
    python stratified_split_yolo.py
"""

import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import yaml

# =========================================================================
# HARD-CODED CONFIG - edit these for your dataset
# =========================================================================
SOURCE_DIR = Path("path/to/my_dataset")       # must contain images/ and labels/
OUTPUT_DIR = Path("path/to/my_dataset_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Set to a list like ["Angry", "Contempt", ...] to hard-code names, or leave
# as None to auto-detect from a classes.txt in SOURCE_DIR / SOURCE_DIR/labels.
CLASS_NAMES = None

SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# =========================================================================


def load_class_names(source_dir: Path, num_classes: int):
    if CLASS_NAMES is not None:
        return CLASS_NAMES
    for candidate in (source_dir / "classes.txt", source_dir / "labels" / "classes.txt"):
        if candidate.is_file():
            names = [l.strip() for l in candidate.read_text(encoding="utf-8").splitlines() if l.strip()]
            if names:
                return names
    return [f"class_{i}" for i in range(num_classes)]


def parse_yolo_label(label_path: Path):
    """Return list of (class_id, line_text) for every valid box line."""
    boxes = []
    if not label_path.is_file():
        return boxes
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        boxes.append((cid, line))
    return boxes


def dominant_class(class_ids):
    """Most frequent class id in the list; ties broken by lowest id."""
    counts = Counter(class_ids)
    max_count = max(counts.values())
    winners = sorted(cid for cid, c in counts.items() if c == max_count)
    return winners[0]


def allocate_group(image_names, train_ratio, val_ratio, test_ratio, rng, class_label, warnings):
    """
    Split one dominant-class group of image names into (train, val, test)
    lists, guaranteeing at least one item in val/test when the group is
    big enough and that ratio is > 0. Handles tiny groups gracefully.
    """
    names = list(image_names)
    rng.shuffle(names)
    n = len(names)

    if n == 1:
        warnings.append(f"Class '{class_label}': only 1 image total -> placed in train only "
                         f"(cannot validate/test this class).")
        return names, [], []

    if n == 2:
        warnings.append(f"Class '{class_label}': only 2 images total -> 1 train / 1 val, "
                         f"none in test (cannot fully evaluate this class in test).")
        return [names[0]], [names[1]], []

    # n >= 3: proportional split, guaranteeing >=1 in val/test if their ratio > 0
    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    if test_ratio > 0:
        n_test = max(n_test, 1)
    if val_ratio > 0:
        n_val = max(n_val, 1)
    # never let val+test consume the whole group - always leave >=1 for train
    while n_val + n_test >= n:
        if n_test >= n_val and n_test > (1 if test_ratio > 0 else 0):
            n_test -= 1
        elif n_val > (1 if val_ratio > 0 else 0):
            n_val -= 1
        else:
            break

    test_names = names[:n_test]
    val_names = names[n_test:n_test + n_val]
    train_names = names[n_test + n_val:]

    if n < 10:
        warnings.append(f"Class '{class_label}': only {n} images total -> split sizes are small "
                         f"({len(train_names)} train / {len(val_names)} val / {len(test_names)} test); "
                         f"metrics on this class will be noisy.")

    return train_names, val_names, test_names


def stratified_yolo_split(source_dir: Path, output_dir: Path,
                           train_ratio: float, val_ratio: float, test_ratio: float,
                           seed: int = 42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
        "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0"

    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(f"'{source_dir}' must contain 'images/' and 'labels/' subfolders.")

    image_files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    rng = random.Random(seed)

    # group image stems by dominant class; images with no boxes go in their own bucket
    groups = defaultdict(list)   # dominant_class_id_or_"background" -> [image_path, ...]
    max_class_id = -1
    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = parse_yolo_label(label_path)
        if not boxes:
            groups["background"].append(img_path)
            continue
        class_ids = [b[0] for b in boxes]
        max_class_id = max(max_class_id, max(class_ids))
        groups[dominant_class(class_ids)].append(img_path)

    class_names = load_class_names(source_dir, max_class_id + 1)

    warnings = []
    split_assignment = {"train": [], "val": [], "test": []}
    for group_label, paths in groups.items():
        label_display = class_names[group_label] if isinstance(group_label, int) and group_label < len(class_names) else str(group_label)
        train_p, val_p, test_p = allocate_group(paths, train_ratio, val_ratio, test_ratio, rng, label_display, warnings)
        split_assignment["train"].extend(train_p)
        split_assignment["val"].extend(val_p)
        split_assignment["test"].extend(test_p)

    # write out files
    split_dir_names = {"train": "train", "val": "valid", "test": "test"}
    for split_key, dir_name in split_dir_names.items():
        img_out = output_dir / dir_name / "images"
        lbl_out = output_dir / dir_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in split_assignment[split_key]:
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.is_file():
                shutil.copy2(label_path, lbl_out / label_path.name)
            else:
                (lbl_out / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

    # per-class instance distribution report (based on actual final splits)
    def count_instances(paths):
        counter = Counter()
        for img_path in paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            for cid, _ in parse_yolo_label(label_path):
                counter[cid] += 1
        return counter

    dist = {split: count_instances(paths) for split, paths in split_assignment.items()}

    # write data.yaml (Roboflow / Ultralytics style)
    data_yaml = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    return {
        "counts": {k: len(v) for k, v in split_assignment.items()},
        "instance_distribution": dist,
        "class_names": class_names,
        "warnings": warnings,
    }


def main():
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Ratios: train={TRAIN_RATIO} val={VAL_RATIO} test={TEST_RATIO}\n")

    result = stratified_yolo_split(
        SOURCE_DIR, OUTPUT_DIR,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
        seed=SEED,
    )

    print("Image counts per split:")
    for split, n in result["counts"].items():
        print(f"  {split:6s}: {n} images")

    print("\nInstance (box) counts per class per split:")
    header = f"  {'class':<15}" + "".join(f"{s:>10}" for s in ("train", "val", "test"))
    print(header)
    for cid, name in enumerate(result["class_names"]):
        row = f"  {name:<15}"
        for split in ("train", "val", "test"):
            row += f"{result['instance_distribution'][split].get(cid, 0):>10}"
        print(row)

    if result["warnings"]:
        print("\nClass-imbalance warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")

    print(f"\nDone. Split dataset + data.yaml written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
