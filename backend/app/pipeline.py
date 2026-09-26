import time
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from .detector import FaceDetector
from .classifier import EmotionClassifier
from .preprocess import validate_and_decode_image

class Pipeline:
    def __init__(self, detector: FaceDetector = None, classifier: EmotionClassifier = None):
        self.detector = detector or FaceDetector()
        self.classifier = classifier or EmotionClassifier()

    def process_bytes(self, image_bytes: bytes, filename: str = "image.jpg", conf_thresh: float = None, margin: float = None) -> Tuple[Dict[str, Any], np.ndarray]:
        start_time = time.time()
        img_bgr = validate_and_decode_image(image_bytes)

        kwargs = {}
        if conf_thresh is not None:
            kwargs["conf_thresh"] = conf_thresh
        if margin is not None:
            kwargs["margin"] = margin

        faces = self.detector.detect(img_bgr, **kwargs)

        detections = []
        annotated_bgr = img_bgr.copy()

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face["bbox"]
            face_crop = img_bgr[y1:y2, x1:x2]
            emo_res = self.classifier.classify(face_crop)

            detection_entry = {
                "face_id": idx,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "face_confidence": round(face["det_conf"], 4),
                "emotion_label": emo_res["label"],
                "emotion_confidence": round(emo_res["cls_conf"], 4),
                "all_scores": emo_res.get("all_scores")
            }
            detections.append(detection_entry)

            # Draw annotations on image
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{emo_res['label']} {emo_res['cls_conf']*100:.1f}%"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated_bgr, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        exec_time_ms = round((time.time() - start_time) * 1000, 2)

        result_payload = {
            "status": "success",
            "filename": filename,
            "total_faces": len(detections),
            "detections": detections,
            "execution_time_ms": exec_time_ms
        }

        return result_payload, annotated_bgr
