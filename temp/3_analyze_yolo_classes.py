#!/usr/bin/env python3
"""
YOLO Class Distribution Analyzer
=================================

A small desktop GUI tool: pick a dataset folder and see a table breaking
down every class - how many bounding-box instances it has, and in how many
distinct images it appears (useful for spotting class imbalance and
rare/under-represented classes before training).

Expected folder structure (you select the PARENT folder in the app):

    my_dataset/
        images/
            img001.jpg
            ...
        labels/
            img001.txt      # YOLO format: <class_id> <xc> <yc> <w> <h>
            ...

Optional: a "classes.txt" file (one class name per line, in class-id order)
in the dataset root or inside labels/ will be used to show class names.

Table columns:
    Class ID        - the numeric class id from the label files
    Class Name      - from classes.txt if available, else same as id
    Instance Count  - total number of bounding boxes of this class across
                      the whole dataset
    Image Count     - number of DISTINCT images that contain at least one
                      box of this class ("available in how many images")
    % of Images     - Image Count / total images with any label, as a %

Also shows overall totals (images, images with no boxes / no label file,
total instances) above the table, and lets you export the table to CSV.

Controls:
    - "Open Folder..." button -> pick the dataset folder and (re)compute
    - Click a column header -> sort by that column
    - "Export CSV..." button -> save the current table to a .csv file

Balanced subset creation:
    Once the table is populated, you can also build a new, class-balanced
    dataset:
        - "Images per class:" field + "Apply to All" -> sets every row's
          "Target Count" to that number (capped at how many images that
          class actually has available).
        - Double-click a row's "Target Count" cell to set a custom number
          for just that class (e.g. give a rare class fewer images than
          the common ones).
        - "Create Dataset..." button -> pick an output folder, and the
          tool copies images + their full label files into a new
          images/ + labels/ folder pair there. Each image is copied only
          ONCE even if it satisfies more than one class's target (an
          object-detection image can contain several classes' boxes), so
          the output has no duplicate image files. If a class doesn't
          have enough distinct images to reach its target, you'll get a
          warning listing exactly how short it was.

Requirements:
    (standard library only - tkinter must be available, which it is on
    most standard Python installs on Windows/macOS; on Linux you may need
    `sudo apt install python3-tk`)

Run:
    python 3_analyze_yolo_classes.py
    (or)
    python 3_analyze_yolo_classes.py /path/to/my_dataset
"""

import csv
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_class_names(dataset_dir: Path):
    candidates = [dataset_dir / "classes.txt", dataset_dir / "labels" / "classes.txt"]
    for c in candidates:
        if c.is_file():
            try:
                names = [line.strip() for line in c.read_text(encoding="utf-8").splitlines()]
                names = [n for n in names if n != ""]
                if names:
                    return names
            except Exception:
                pass
    return None


def parse_yolo_label(label_path: Path):
    """Return list of class ids found in this label file (one per box, may repeat)."""
    class_ids = []
    if not label_path.is_file():
        return class_ids
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError:
            continue
    return class_ids


def analyze_dataset(dataset_dir: Path):
    """
    Returns a dict with:
        rows: list of dicts {class_id, class_name, instance_count, image_count, pct_images}
        total_images: int
        images_with_labels: int
        images_without_boxes: int   (label file missing, empty, or unparsable)
        total_instances: int
        images_containing: dict {class_id: sorted list of image stems}
        stem_to_path: dict {image stem: Path to that image file}
        labels_dir: Path
    Raises ValueError if the folder structure is invalid.
    """
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError(
            f"'{dataset_dir}' must contain both an 'images' and a 'labels' subfolder.\n"
            f"Looked for:\n  {images_dir}\n  {labels_dir}"
        )

    image_files = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")

    class_names = load_class_names(dataset_dir)

    instance_count = defaultdict(int)      # class_id -> total boxes
    images_containing = defaultdict(set)   # class_id -> set of image stems
    stem_to_path = {}                      # image stem -> Path
    images_without_boxes = 0

    for img_path in image_files:
        stem_to_path[img_path.stem] = img_path
        label_path = labels_dir / (img_path.stem + ".txt")
        class_ids = parse_yolo_label(label_path)
        if not class_ids:
            images_without_boxes += 1
            continue
        for cid in class_ids:
            instance_count[cid] += 1
        for cid in set(class_ids):
            images_containing[cid].add(img_path.stem)

    total_images = len(image_files)
    images_with_labels = total_images - images_without_boxes
    total_instances = sum(instance_count.values())

    all_class_ids = sorted(set(instance_count.keys()) | set(images_containing.keys()))

    rows = []
    for cid in all_class_ids:
        name = str(cid)
        if class_names and 0 <= cid < len(class_names):
            name = class_names[cid]
        img_count = len(images_containing[cid])
        pct = (img_count / images_with_labels * 100) if images_with_labels else 0.0
        rows.append({
            "class_id": cid,
            "class_name": name,
            "instance_count": instance_count[cid],
            "image_count": img_count,
            "pct_images": round(pct, 1),
        })

    return {
        "rows": rows,
        "total_images": total_images,
        "images_with_labels": images_with_labels,
        "images_without_boxes": images_without_boxes,
        "total_instances": total_instances,
        "images_containing": {cid: sorted(stems) for cid, stems in images_containing.items()},
        "stem_to_path": stem_to_path,
        "labels_dir": labels_dir,
    }


def write_yolo_yaml(output_dir: Path, class_names):
    """
    Write data.yaml required for YOLO model training.
    class_names can be a dict {class_id: name} or list [name0, name1, ...]
    """
    if isinstance(class_names, list):
        class_map = {i: name for i, name in enumerate(class_names)}
    elif isinstance(class_names, dict):
        class_map = class_names
    else:
        class_map = {}

    yaml_path = output_dir / "data.yaml"
    lines = [
        "# YOLO Dataset Configuration",
        "path: .",
        "train: images",
        "val: images",
        "",
        f"nc: {len(class_map)}",
        "names:",
    ]
    for cid in sorted(class_map.keys()):
        name = class_map[cid]
        clean_name = str(name).replace('"', '\\"')
        lines.append(f'  {cid}: "{clean_name}"')

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def create_balanced_subset(images_containing, stem_to_path, labels_dir: Path,
                            class_targets: dict, output_dir: Path, seed: int = 42,
                            class_names=None):
    """
    Build a new images/ + labels/ folder pair containing, for each class id
    in class_targets, that many DISTINCT images which contain that class.
    Also creates a data.yaml file required for YOLO model training.

    Images are shared across classes when possible: if an image already
    selected (because it satisfies some other class) also contains this
    class, it counts toward this class's target too instead of being
    skipped or duplicated - so every image is copied at most once.

    Rarer classes (fewer available images) are processed first so their
    limited pool of images gets allocated before common classes have a
    chance to "use up" the shared images list (this mostly affects which
    extra images get pulled in, not correctness).

    Returns (selected_stems: set, achieved: dict{class_id: int}, warnings: list[str])
    """
    rng = random.Random(seed)
    selected = set()
    achieved = {}
    warnings = []

    # process classes with the smallest available pool first
    order = sorted(class_targets.keys(), key=lambda c: len(images_containing.get(c, [])))

    for cid in order:
        target = int(class_targets.get(cid, 0) or 0)
        available = list(images_containing.get(cid, []))
        if target <= 0:
            achieved[cid] = 0
            continue

        rng.shuffle(available)
        already_selected = [s for s in available if s in selected]
        if len(already_selected) >= target:
            achieved[cid] = target
            continue

        need = target - len(already_selected)
        candidates = [s for s in available if s not in selected]
        take = candidates[:need]
        for s in take:
            selected.add(s)
        achieved[cid] = len(already_selected) + len(take)

        if achieved[cid] < target:
            warnings.append(
                f"Class {cid}: requested {target} images but only {achieved[cid]} distinct "
                f"images are available for this class (dataset has no more)."
            )

    # copy files
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for stem in selected:
        img_path = stem_to_path.get(stem)
        if img_path is None or not img_path.is_file():
            continue
        shutil.copy2(img_path, images_out / img_path.name)
        label_path = labels_dir / (stem + ".txt")
        if label_path.is_file():
            shutil.copy2(label_path, labels_out / label_path.name)

    # write data.yaml for YOLO training
    if class_names is None:
        class_names = {cid: str(cid) for cid in class_targets.keys()}
    write_yolo_yaml(output_dir, class_names)

    # write classes.txt if missing
    classes_file = output_dir / "classes.txt"
    if not classes_file.is_file() and class_names:
        if isinstance(class_names, dict):
            max_cid = max(class_names.keys()) if class_names else -1
            lines = [str(class_names.get(i, i)) for i in range(max_cid + 1)]
        else:
            lines = [str(n) for n in class_names]
        classes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return selected, achieved, warnings


class ClassAnalyzerApp(tk.Tk):
    COLUMNS = [
        ("class_id", "Class ID", 70),
        ("class_name", "Class Name", 180),
        ("instance_count", "Instance Count", 120),
        ("image_count", "Image Count", 110),
        ("pct_images", "% of Images", 100),
        ("target_count", "Target Count", 110),
    ]

    def __init__(self, initial_folder=None):
        super().__init__()
        self.title("YOLO Class Distribution Analyzer")
        self.geometry("820x640")
        self.minsize(600, 440)

        self.dataset_dir = None
        self.current_rows = []
        self.sort_state = {}  # col -> ascending bool
        self.analysis_result = None  # full result dict from analyze_dataset

        self._build_ui()

        if initial_folder:
            self.analyze_folder(Path(initial_folder))

    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Folder...", command=self.open_folder).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Export CSV...", command=self.export_csv).pack(side=tk.LEFT, padx=(8, 0))

        self.summary_var = tk.StringVar(value="Open a dataset folder to analyze its classes.")
        ttk.Label(self, textvariable=self.summary_var, padding=(8, 4), justify=tk.LEFT,
                  wraplength=740).pack(side=tk.TOP, fill=tk.X)

        table_frame = ttk.Frame(self, padding=(8, 0, 8, 8))
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        col_ids = [c[0] for c in self.COLUMNS]
        self.tree = ttk.Treeview(table_frame, columns=col_ids, show="headings", selectmode="browse")
        for col_id, heading, width in self.COLUMNS:
            self.tree.heading(col_id, text=heading, command=lambda c=col_id: self.sort_by(c))
            anchor = tk.W if col_id == "class_name" else tk.CENTER
            self.tree.column(col_id, width=width, anchor=anchor)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.tag_configure("low", background="#ffe5e5")   # highlight under-represented classes
        self.tree.bind("<Double-1>", self._on_cell_double_click)

        # ---- balanced-subset creation controls ----
        balance_frame = ttk.LabelFrame(self, text="Create a class-balanced dataset", padding=8)
        balance_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))

        ttk.Label(balance_frame, text="Images per class:").pack(side=tk.LEFT)
        self.target_entry_var = tk.StringVar(value="")
        ttk.Entry(balance_frame, textvariable=self.target_entry_var, width=8).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(balance_frame, text="Apply to All", command=self.apply_target_to_all).pack(side=tk.LEFT)
        ttk.Label(balance_frame, text="   (double-click a row's Target Count cell to set one class individually)",
                  foreground="#555555").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(balance_frame, text="Create Dataset...", command=self.create_dataset).pack(side=tk.RIGHT)

    def open_folder(self):
        folder = filedialog.askdirectory(title="Select dataset folder (must contain images/ and labels/)")
        if folder:
            self.analyze_folder(Path(folder))

    def analyze_folder(self, folder: Path):
        try:
            result = analyze_dataset(folder)
        except ValueError as e:
            messagebox.showerror("Invalid folder", str(e))
            return

        self.dataset_dir = folder
        self.title(f"YOLO Class Distribution Analyzer - {folder}")
        self.analysis_result = result
        self.current_rows = result["rows"]
        # default target = all available images for that class; user can edit down
        for r in self.current_rows:
            r["target_count"] = r["image_count"]

        max_img_count = max((r["image_count"] for r in self.current_rows), default=0)
        self.summary_var.set(
            f"Dataset: {folder}\n"
            f"Total images: {result['total_images']}   |   "
            f"Images with at least one box: {result['images_with_labels']}   |   "
            f"Images with no boxes/label: {result['images_without_boxes']}   |   "
            f"Total box instances: {result['total_instances']}   |   "
            f"Classes found: {len(self.current_rows)}\n"
            f"(Rows shaded red have less than 20% of the image-coverage of the most common class - "
            f"possible class imbalance.)"
        )

        self.populate_table(self.current_rows)

    def populate_table(self, rows):
        self.tree.delete(*self.tree.get_children())
        max_img_count = max((r["image_count"] for r in rows), default=0)
        for r in rows:
            is_low = max_img_count > 0 and r["image_count"] < 0.2 * max_img_count
            tags = ("low",) if is_low else ()
            self.tree.insert("", tk.END, values=(
                r["class_id"], r["class_name"], r["instance_count"], r["image_count"],
                f"{r['pct_images']}%", r.get("target_count", 0)
            ), tags=tags)

    def _on_cell_double_click(self, event):
        if not self.current_rows:
            return
        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)  # e.g. "#6"
        if not row_id or col_id != "#6":  # target_count is the 6th column
            return
        item = self.tree.item(row_id)
        class_id = item["values"][0]
        row = next((r for r in self.current_rows if r["class_id"] == class_id), None)
        if row is None:
            return

        new_val = simpledialog.askinteger(
            "Set target count",
            f"How many images of class '{row['class_name']}' (id {row['class_id']}) "
            f"should the balanced dataset include?\n"
            f"(Max available for this class: {row['image_count']})",
            initialvalue=row.get("target_count", row["image_count"]),
            minvalue=0,
        )
        if new_val is None:
            return
        if new_val > row["image_count"]:
            messagebox.showwarning(
                "Not enough images",
                f"Only {row['image_count']} distinct images contain class '{row['class_name']}'. "
                f"Target has been capped to {row['image_count']}."
            )
            new_val = row["image_count"]
        row["target_count"] = new_val
        self.populate_table(self.current_rows)

    def apply_target_to_all(self):
        if not self.current_rows:
            messagebox.showinfo("No data", "Open a dataset folder first.")
            return
        raw = self.target_entry_var.get().strip()
        if not raw.isdigit():
            messagebox.showerror("Invalid number", "Enter a whole number in 'Images per class'.")
            return
        target = int(raw)
        for r in self.current_rows:
            r["target_count"] = min(target, r["image_count"])
        self.populate_table(self.current_rows)

    def create_dataset(self):
        if not self.current_rows or not self.analysis_result:
            messagebox.showinfo("No data", "Open a dataset folder first.")
            return

        class_targets = {r["class_id"]: r.get("target_count", 0) for r in self.current_rows}
        if all(v <= 0 for v in class_targets.values()):
            messagebox.showinfo("Nothing to create", "Every class has a target of 0 - nothing to copy.")
            return

        output_folder = filedialog.askdirectory(title="Choose (or create) an output folder for the new dataset")
        if not output_folder:
            return
        output_dir = Path(output_folder)

        if any(output_dir.iterdir()) if output_dir.is_dir() else False:
            if not messagebox.askyesno(
                "Folder not empty",
                f"'{output_dir}' is not empty. Files may be added alongside existing ones. Continue?"
            ):
                return

        class_names_map = {r["class_id"]: r["class_name"] for r in self.current_rows}

        selected, achieved, warnings = create_balanced_subset(
            images_containing=self.analysis_result["images_containing"],
            stem_to_path=self.analysis_result["stem_to_path"],
            labels_dir=self.analysis_result["labels_dir"],
            class_targets=class_targets,
            output_dir=output_dir,
            class_names=class_names_map,
        )

        # copy classes.txt along, if we found one, so the new dataset is self-describing
        classes_src = None
        if self.dataset_dir:
            for c in (self.dataset_dir / "classes.txt", self.dataset_dir / "labels" / "classes.txt"):
                if c.is_file():
                    classes_src = c
                    break
        if classes_src:
            shutil.copy2(classes_src, output_dir / "classes.txt")

        summary_lines = [f"Created dataset with {len(selected)} unique images at:\n{output_dir}\n"]
        summary_lines.append("Files created:")
        summary_lines.append("  - images/")
        summary_lines.append("  - labels/")
        summary_lines.append("  - data.yaml (YOLO training config)")
        if (output_dir / "classes.txt").is_file():
            summary_lines.append("  - classes.txt")
        summary_lines.append("\nAchieved counts per class:")
        for r in self.current_rows:
            cid = r["class_id"]
            if class_targets.get(cid, 0) > 0:
                summary_lines.append(f"  {r['class_name']} (id {cid}): {achieved.get(cid, 0)} / {class_targets[cid]}")
        if warnings:
            summary_lines.append("\nWarnings:")
            summary_lines.extend(f"  - {w}" for w in warnings)

        messagebox.showinfo("Dataset created", "\n".join(summary_lines))

    def sort_by(self, col):
        ascending = self.sort_state.get(col, True)
        self.current_rows.sort(key=lambda r: r[col], reverse=not ascending)
        self.sort_state[col] = not ascending
        self.populate_table(self.current_rows)

    def export_csv(self):
        if not self.current_rows:
            messagebox.showinfo("Nothing to export", "Open a dataset folder first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Class ID", "Class Name", "Instance Count", "Image Count", "% of Images", "Target Count"])
            for r in self.current_rows:
                writer.writerow([r["class_id"], r["class_name"], r["instance_count"],
                                  r["image_count"], r["pct_images"], r.get("target_count", 0)])
        messagebox.showinfo("Exported", f"Table exported to:\n{path}")


def main():
    initial_folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = ClassAnalyzerApp(initial_folder=initial_folder)
    app.mainloop()


if __name__ == "__main__":
    main()
