#!/usr/bin/env python3
"""
YOLO Dataset Cleaner
====================

Removes corrupted images and duplicate images from a YOLO-format dataset,
ALWAYS deleting the matching label file together with its image so the
images/ and labels/ folders stay in sync.

Expected folder structure (you pass the PARENT folder path):

    my_dataset/
        images/
            img001.jpg
            img002.png
            ...
        labels/
            img001.txt
            img002.txt
            ...

What it does, in order:
    1. CORRUPTED CHECK
       Tries to open + fully load every image with Pillow. Anything that
       fails (truncated file, 0 bytes, unreadable header, etc.) is flagged
       as corrupted.

    2. DUPLICATE CHECK (on the remaining, valid images)
       a) Exact duplicates: identical file bytes (MD5 hash match).
       b) Near-duplicates (optional, on by default): a perceptual hash
          (dHash) that catches images that are pixel-identical in content
          but re-saved/re-encoded (different compression, resave, etc.).
          Use --exact-only to disable this and only catch byte-identical
          files.
       The FIRST occurrence (alphabetical by filename) of each duplicate
       group is kept; later ones are removed.

    3. For every removed image, the matching label file
       (images/<name>.<ext> -> labels/<name>.txt) is removed too, if it
       exists. Orphan labels are never removed unless their image is also
       being removed.

Note on the perceptual hash: it looks at relative brightness changes between
neighboring pixels, so it can occasionally flag two images that are both
almost perfectly flat/solid-colored (e.g. two different plain backgrounds)
as "near-duplicates" even though their color differs. If your dataset has
many flat/plain images and you see suspicious matches, re-run with
--exact-only to only remove byte-identical files.

By default this does a DRY RUN and only prints what it *would* delete.
Pass --apply to actually delete the files. A summary + a text log of
everything removed is printed at the end (and optionally saved to a file).

Usage:
    python 2_clean_yolo_dataset.py "D:\Ranii\Facial Expression ClassificationDetection_Dataset\test"                # dry run
    python 2_clean_yolo_dataset.py "D:\Ranii\Facial Expression ClassificationDetection_Dataset\test" --apply         # actually delete
    python 2_clean_yolo_dataset.py "D:\Ranii\Facial Expression ClassificationDetection_Dataset\test" --apply --exact-only
    python 2_clean_yolo_dataset.py "D:\Ranii\Facial Expression ClassificationDetection_Dataset\train" --apply --log removed_files.txt

Requirements:
    pip install pillow
"""

import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_label_for_image(labels_dir: Path, image_path: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


def is_corrupted(image_path: Path) -> bool:
    """Returns True if the image cannot be opened/decoded fully."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # checks structure, doesn't decode pixel data
        # verify() leaves the file in a state where it can't be reused,
        # so re-open for a full decode pass to also catch truncated pixel data.
        with Image.open(image_path) as img:
            img.load()
        return False
    except Exception:
        return True


def dhash(image_path: Path, hash_size: int = 8):
    """Simple, dependency-free perceptual hash (difference hash).
    Returns a binary string, or None if the image can't be read."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
            pixels = list(img.getdata())
            bits = []
            for row in range(hash_size):
                row_start = row * (hash_size + 1)
                for col in range(hash_size):
                    left = pixels[row_start + col]
                    right = pixels[row_start + col + 1]
                    bits.append("1" if left > right else "0")
            return "".join(bits)
    except Exception:
        return None


def hamming_distance(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Remove corrupted and duplicate image/label pairs from a YOLO dataset."
    )
    parser.add_argument("folder", help="Path to the dataset folder (must contain images/ and labels/)")
    parser.add_argument("--apply", action="store_true",
                         help="Actually delete files. Without this flag, it's a dry run (report only).")
    parser.add_argument("--exact-only", action="store_true",
                         help="Only remove byte-identical duplicates; skip perceptual near-duplicate check.")
    parser.add_argument("--phash-threshold", type=int, default=0,
                         help="Hamming distance <= this value counts as a near-duplicate (default 0 = only "
                              "exact perceptual match). Raise cautiously, e.g. 2-4, to catch slightly modified "
                              "re-saves; higher values risk flagging genuinely different images.")
    parser.add_argument("--log", metavar="FILE",
                         help="Optional path to write a text log of every removed file.")
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    images_dir = root / "images"
    labels_dir = root / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"ERROR: '{root}' must contain both an 'images' and a 'labels' subfolder.")
        print(f"  Looked for: {images_dir}")
        print(f"              {labels_dir}")
        sys.exit(1)

    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not image_files:
        print(f"No image files found in {images_dir}. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(image_files)} images in {images_dir}\n")

    removal_log = []  # list of (image_path, label_path_or_None, reason)

    # ---------- Step 1: corrupted check ----------
    print("Step 1/2: checking for corrupted images...")
    valid_images = []
    for img_path in image_files:
        if is_corrupted(img_path):
            label_path = find_label_for_image(labels_dir, img_path)
            removal_log.append((img_path, label_path if label_path.is_file() else None, "corrupted"))
        else:
            valid_images.append(img_path)
    print(f"  -> {len(removal_log)} corrupted image(s) found.\n")

    # ---------- Step 2: duplicate check ----------
    print("Step 2/2: checking for duplicate images...")
    seen_md5 = {}       # md5 -> kept image path
    seen_phash = {}      # phash -> kept image path
    kept_images = []

    for img_path in valid_images:
        md5 = md5_of_file(img_path)
        if md5 in seen_md5:
            label_path = find_label_for_image(labels_dir, img_path)
            removal_log.append(
                (img_path, label_path if label_path.is_file() else None,
                 f"exact duplicate of {seen_md5[md5].name}")
            )
            continue

        is_dup = False
        if not args.exact_only:
            ph = dhash(img_path)
            if ph is not None:
                for existing_hash, existing_path in seen_phash.items():
                    if hamming_distance(ph, existing_hash) <= args.phash_threshold:
                        label_path = find_label_for_image(labels_dir, img_path)
                        removal_log.append(
                            (img_path, label_path if label_path.is_file() else None,
                             f"near-duplicate of {existing_path.name}")
                        )
                        is_dup = True
                        break
                if not is_dup:
                    seen_phash[ph] = img_path

        if is_dup:
            continue

        seen_md5[md5] = img_path
        kept_images.append(img_path)

    dup_count = len(removal_log) - sum(1 for _, _, r in removal_log if r == "corrupted")
    print(f"  -> {dup_count} duplicate image(s) found.\n")

    # ---------- Report / apply ----------
    if not removal_log:
        print("Dataset is clean: no corrupted or duplicate files found.")
        return

    print(f"{'Would remove' if not args.apply else 'Removing'} {len(removal_log)} image/label pair(s):\n")
    log_lines = []
    for img_path, label_path, reason in removal_log:
        line = f"  [{reason}] {img_path.relative_to(root)}" + (
            f"  +  {label_path.relative_to(root)}" if label_path else "  (no matching label file)"
        )
        print(line)
        log_lines.append(line)

    if args.apply:
        removed_images = removed_labels = 0
        for img_path, label_path, _reason in removal_log:
            try:
                img_path.unlink()
                removed_images += 1
            except FileNotFoundError:
                pass
            if label_path:
                try:
                    label_path.unlink()
                    removed_labels += 1
                except FileNotFoundError:
                    pass
        print(f"\nDone. Deleted {removed_images} image(s) and {removed_labels} label(s).")
        remaining = len(image_files) - removed_images
        print(f"Remaining images: {remaining}")
    else:
        print(f"\nDRY RUN - no files were deleted. Re-run with --apply to actually delete these {len(removal_log)} pair(s).")

    if args.log:
        Path(args.log).write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
