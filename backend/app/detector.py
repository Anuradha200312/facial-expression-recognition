import os
import numpy as np
from typing import List, Dict, Any
from .config import FACE_MODEL_PATH, FACE_CONF_THRESH, FACE_CROP_MARGIN
from .preprocess import expand_bbox

class FaceDetector:
    def __init__(self, model_path: str = FACE_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Failed to load YOLO face model from {self.model_path}: {e}")
                self.is_loaded = False
        else:
            print(f"Notice: Face model file not found at {self.model_path}. Detector running in mock mode.")

    def detect(self, image_bgr: np.ndarray, conf_thresh: float = FACE_CONF_THRESH, margin: float = FACE_CROP_MARGIN) -> List[Dict[str, Any]]:
        if image_bgr is None or image_bgr.size == 0:
            return []

        img_h, img_w = image_bgr.shape[:2]
        faces = []

        if self.is_loaded and self.model is not None:
            results = self.model.predict(image_bgr, conf=conf_thresh, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    ex_x1, ex_y1, ex_x2, ex_y2 = expand_bbox(int(x1), int(y1), int(x2), int(y2), img_w, img_h, margin)
                    faces.append({
                        "bbox": (ex_x1, ex_y1, ex_x2, ex_y2),
                        "det_conf": float(box.conf[0])
                    })
        else:
            # Fallback for testing/mock when model file is not yet uploaded
            # Detects a central face if image dimensions are adequate
            if img_w >= 40 and img_h >= 40:
                cx, cy = img_w // 2, img_h // 2
                w_half, h_half = int(img_w * 0.25), int(img_h * 0.25)
                x1, y1 = max(0, cx - w_half), max(0, cy - h_half)
                x2, y2 = min(img_w, cx + w_half), min(img_h, cy + h_half)
                faces.append({
                    "bbox": (x1, y1, x2, y2),
                    "det_conf": 0.95
                })

        return faces
