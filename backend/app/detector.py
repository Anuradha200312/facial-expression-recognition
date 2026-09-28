import os
import numpy as np
from typing import List, Dict, Any, Tuple
from .config import FACE_MODEL_PATH, FACE_CONF_THRESH, FACE_CROP_MARGIN
from .preprocess import expand_bbox

def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class CentroidFaceTracker:
    """Persistent face tracker based on IoU and centroid matching across frames."""
    def __init__(self, iou_thresh: float = 0.3, max_disappeared: int = 15):
        self.next_id = 1
        self.tracks = {}  # track_id -> {"bbox": (x1, y1, x2, y2), "disappeared": int}
        self.iou_thresh = iou_thresh
        self.max_disappeared = max_disappeared

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
            return []

        if not self.tracks:
            updated_detections = []
            for det in detections:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "disappeared": 0}
                det_copy = dict(det)
                det_copy["track_id"] = tid
                updated_detections.append(det_copy)
            return updated_detections

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid]["bbox"] for tid in track_ids]

        assigned_det = set()
        assigned_track = set()
        updated_detections = [None] * len(detections)

        # Build IoU matrix
        iou_matrix = np.zeros((len(detections), len(track_ids)), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, t_box in enumerate(track_boxes):
                iou_matrix[i, j] = compute_iou(det["bbox"], t_box)

        # Match highest IoU pairs
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.ravel())
            for idx in flat_indices:
                d_idx, t_idx = divmod(idx, len(track_ids))
                iou_val = iou_matrix[d_idx, t_idx]
                if iou_val < self.iou_thresh:
                    break
                if d_idx in assigned_det or t_idx in assigned_track:
                    continue

                tid = track_ids[t_idx]
                self.tracks[tid] = {"bbox": detections[d_idx]["bbox"], "disappeared": 0}
                det_copy = dict(detections[d_idx])
                det_copy["track_id"] = tid
                updated_detections[d_idx] = det_copy

                assigned_det.add(d_idx)
                assigned_track.add(t_idx)

        # Unassigned detections get new track IDs
        for i, det in enumerate(detections):
            if i not in assigned_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": det["bbox"], "disappeared": 0}
                det_copy = dict(det)
                det_copy["track_id"] = tid
                updated_detections[i] = det_copy

        # Unassigned tracks increase disappeared counter
        for j, tid in enumerate(track_ids):
            if j not in assigned_track:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        return updated_detections

    def reset(self):
        self.next_id = 1
        self.tracks = {}

class FaceDetector:
    def __init__(self, model_path: str = FACE_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.tracker = CentroidFaceTracker()
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
                    det_conf = float(box.conf[0])
                    if det_conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    ex_x1, ex_y1, ex_x2, ex_y2 = expand_bbox(int(x1), int(y1), int(x2), int(y2), img_w, img_h, margin)
                    faces.append({
                        "bbox": (ex_x1, ex_y1, ex_x2, ex_y2),
                        "det_conf": det_conf
                    })
        else:
            # Fallback for testing/mock when model file is not yet uploaded
            if img_w >= 40 and img_h >= 40:
                cx, cy = img_w // 2, img_h // 2
                w_half, h_half = int(img_w * 0.25), int(img_h * 0.25)
                x1, y1 = max(0, cx - w_half), max(0, cy - h_half)
                x2, y2 = min(img_w, cx + w_half), min(img_h, cy + h_half)
                if 0.95 >= conf_thresh:
                    faces.append({
                        "bbox": (x1, y1, x2, y2),
                        "det_conf": 0.95
                    })

        return faces

    def track(self, image_bgr: np.ndarray, conf_thresh: float = FACE_CONF_THRESH, margin: float = FACE_CROP_MARGIN) -> List[Dict[str, Any]]:
        faces = self.detect(image_bgr, conf_thresh=conf_thresh, margin=margin)
        return self.tracker.update(faces)

    def reset_tracker(self):
        self.tracker.reset()
