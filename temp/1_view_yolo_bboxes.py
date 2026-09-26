#!/usr/bin/env python3
"""
YOLO Bounding Box Viewer
=========================

A small desktop GUI tool to browse a YOLO-format dataset and visualize the
bounding boxes drawn from the label files.

Expected folder structure (you select the PARENT folder in the app):

    my_dataset/
        images/
            img001.jpg
            img002.png
            ...
        labels/
            img001.txt
            img002.txt
            ...

Each label .txt file has one bounding box per line, in YOLO format:

    <class_id> <x_center> <y_center> <width> <height>

where x_center, y_center, width, height are all normalized (0-1) relative to
the image width/height.

Optional: if a file named "classes.txt" (one class name per line, in the
same order as the class ids) exists inside the selected folder OR inside the
labels/ folder, it will be used to show class names instead of raw ids.

Controls:
    - "Open Folder..." button (or Ctrl+O)   -> pick the dataset folder
    - Left / Right arrow keys, or Prev/Next buttons -> navigate images
    - Home / End                            -> jump to first / last image
    - Mouse wheel                           -> also navigates images

Requirements:
    pip install pillow

Run:
    python 1_view_yolo_bboxes.py
    (or)
    python view_yolo_bboxes.py /path/to/my_dataset
"""

import sys
import colorsys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk, ImageDraw, ImageFont

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_class_names(dataset_dir: Path):
    """Look for a classes.txt in the dataset root or inside labels/."""
    candidates = [
        dataset_dir / "classes.txt",
        dataset_dir / "labels" / "classes.txt",
    ]
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
    """Return a list of (class_id, xc, yc, w, h) floats, normalized 0-1."""
    boxes = []
    if not label_path.is_file():
        return boxes
    for lineno, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = (float(v) for v in parts[1:5])
            boxes.append((cls_id, xc, yc, w, h))
        except ValueError:
            # skip malformed line, but don't crash the viewer
            continue
    return boxes


def class_color(class_id: int):
    """Deterministic, visually distinct color per class id."""
    hue = (class_id * 0.61803398875) % 1.0  # golden ratio spread
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


class YoloViewer(tk.Tk):
    def __init__(self, initial_folder=None):
        super().__init__()
        self.title("YOLO Bounding Box Viewer")
        self.geometry("1100x750")
        self.minsize(600, 400)

        self.dataset_dir = None
        self.images_dir = None
        self.labels_dir = None
        self.image_files = []
        self.index = 0
        self.class_names = None
        self.tk_image = None  # keep reference to avoid garbage collection

        self._build_ui()
        self.bind("<Left>", lambda e: self.show_prev())
        self.bind("<Right>", lambda e: self.show_next())
        self.bind("<Home>", lambda e: self.jump_to(0))
        self.bind("<End>", lambda e: self.jump_to(len(self.image_files) - 1))
        self.bind("<Control-o>", lambda e: self.open_folder())
        self.bind("<MouseWheel>", self._on_mousewheel)      # Windows / macOS
        self.bind("<Button-4>", lambda e: self.show_prev())  # Linux scroll up
        self.bind("<Button-5>", lambda e: self.show_next())  # Linux scroll down

        if initial_folder:
            self.load_dataset(Path(initial_folder))

    # ---------- UI ----------
    def _build_ui(self):
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open Folder...", command=self.open_folder).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="<< Prev", command=self.show_prev).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(toolbar, text="Next >>", command=self.show_next).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Open a dataset folder to begin.")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=16)

        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        info_bar = ttk.Frame(self, padding=(6, 4))
        info_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.info_var = tk.StringVar(value="")
        ttk.Label(info_bar, textvariable=self.info_var, wraplength=1050, justify=tk.LEFT).pack(side=tk.LEFT)

    # ---------- Dataset loading ----------
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select dataset folder (must contain images/ and labels/)")
        if folder:
            self.load_dataset(Path(folder))

    def load_dataset(self, folder: Path):
        images_dir = folder / "images"
        labels_dir = folder / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            messagebox.showerror(
                "Invalid folder",
                f"The selected folder must contain an 'images' subfolder and a 'labels' subfolder.\n\n"
                f"Looked for:\n  {images_dir}\n  {labels_dir}",
            )
            return

        image_files = sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not image_files:
            messagebox.showwarning("No images found", f"No image files found in:\n{images_dir}")
            return

        self.dataset_dir = folder
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = image_files
        self.index = 0
        self.class_names = load_class_names(folder)
        self.title(f"YOLO Bounding Box Viewer - {folder}")
        self.show_current()

    # ---------- Navigation ----------
    def show_prev(self):
        if not self.image_files:
            return
        self.index = (self.index - 1) % len(self.image_files)
        self.show_current()

    def show_next(self):
        if not self.image_files:
            return
        self.index = (self.index + 1) % len(self.image_files)
        self.show_current()

    def jump_to(self, idx):
        if not self.image_files:
            return
        self.index = max(0, min(idx, len(self.image_files) - 1))
        self.show_current()

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self.show_prev()
        else:
            self.show_next()

    # ---------- Rendering ----------
    def show_current(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.index]
        label_path = self.labels_dir / (img_path.stem + ".txt")

        try:
            self._current_pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            self.status_var.set(f"[{self.index + 1}/{len(self.image_files)}] FAILED TO OPEN: {img_path.name}")
            self.info_var.set(str(e))
            self.canvas.delete("all")
            return

        self._current_boxes = parse_yolo_label(label_path)
        has_label = label_path.is_file()

        self.status_var.set(f"[{self.index + 1}/{len(self.image_files)}]  {img_path.name}")
        class_summary = ", ".join(
            self._class_label(c) for c in sorted(set(b[0] for b in self._current_boxes))
        ) or "(no boxes)"
        label_note = "" if has_label else "  [!] no label file found"
        self.info_var.set(f"Boxes: {len(self._current_boxes)}  |  Classes: {class_summary}{label_note}")

        self.redraw()

    def _class_label(self, class_id):
        if self.class_names and 0 <= class_id < len(self.class_names):
            return f"{class_id}:{self.class_names[class_id]}"
        return str(class_id)

    def redraw(self):
        if not getattr(self, "_current_pil_image", None):
            return

        canvas_w = max(self.canvas.winfo_width(), 1)
        canvas_h = max(self.canvas.winfo_height(), 1)
        img = self._current_pil_image
        img_w, img_h = img.size

        # scale to fit canvas, preserving aspect ratio
        scale = min(canvas_w / img_w, canvas_h / img_h)
        scale = max(scale, 0.01)
        disp_w, disp_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))

        display_img = img.resize((disp_w, disp_h), Image.LANCZOS)
        draw = ImageDraw.Draw(display_img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        for class_id, xc, yc, w, h in self._current_boxes:
            # convert normalized YOLO coords -> pixel coords in the DISPLAYED image
            box_w = w * disp_w
            box_h = h * disp_h
            cx = xc * disp_w
            cy = yc * disp_h
            x1 = cx - box_w / 2
            y1 = cy - box_h / 2
            x2 = cx + box_w / 2
            y2 = cy + box_h / 2

            color = class_color(class_id)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            label_text = self._class_label(class_id)
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            ty1 = max(0, y1 - text_h - 4)
            draw.rectangle([x1, ty1, x1 + text_w + 4, ty1 + text_h + 4], fill=color)
            draw.text((x1 + 2, ty1 + 1), label_text, fill=(0, 0, 0), font=font)

        self.tk_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        offset_x = (canvas_w - disp_w) // 2
        offset_y = (canvas_h - disp_h) // 2
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.tk_image)


def main():
    initial_folder = sys.argv[1] if len(sys.argv) > 1 else None
    app = YoloViewer(initial_folder=initial_folder)
    app.mainloop()


if __name__ == "__main__":
    main()
