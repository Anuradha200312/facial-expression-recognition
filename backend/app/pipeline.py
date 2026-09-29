import base64
import time
import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List
from .detector import FaceDetector
from .classifier import EmotionClassifier
from .preprocess import validate_and_decode_image, expand_bbox
from .config import DEFAULT_VIDEO_FRAME_STRIDE, FACE_CONF_THRESH, EMOTION_CONF_THRESH

class Pipeline:
    def __init__(self, detector: FaceDetector = None, classifier: EmotionClassifier = None):
        self.detector = detector or FaceDetector()
        self.classifier = classifier or EmotionClassifier()

    def process_bytes(
        self,
        image_bytes: bytes,
        filename: str = "image.jpg",
        conf_thresh: float = None,
        emotion_conf_thresh: float = None,
        margin: float = None,
        is_stream: bool = False
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        start_time = time.time()
        img_bgr = validate_and_decode_image(image_bytes)

        if conf_thresh is None:
            conf_thresh = FACE_CONF_THRESH
        if emotion_conf_thresh is None:
            emotion_conf_thresh = EMOTION_CONF_THRESH

        if is_stream:
            faces = self.detector.track(img_bgr, conf_thresh=conf_thresh, margin=margin)
        else:
            faces = self.detector.detect(img_bgr, conf_thresh=conf_thresh, margin=margin)

        detections = []
        annotated_bgr = img_bgr.copy()

        for idx, face in enumerate(faces):
            if face["det_conf"] < conf_thresh:
                continue
            x1, y1, x2, y2 = face["bbox"]
            face_crop = img_bgr[y1:y2, x1:x2]
            emo_res = self.classifier.classify(face_crop)

            if emo_res["cls_conf"] < emotion_conf_thresh:
                continue

            track_id = face.get("track_id", idx + 1)

            detection_entry = {
                "face_id": idx,
                "track_id": track_id,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "face_confidence": round(face["det_conf"], 4),
                "emotion_label": emo_res["label"],
                "emotion_confidence": round(emo_res["cls_conf"], 4),
                "all_scores": emo_res.get("all_scores")
            }
            detections.append(detection_entry)

            # Draw annotations on image
            cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"ID:{track_id} {emo_res['label']} (Face:{face['det_conf']*100:.0f}%, Emo:{emo_res['cls_conf']*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_bgr, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(annotated_bgr, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        exec_time_ms = round((time.time() - start_time) * 1000, 2)

        result_payload = {
            "status": "success",
            "filename": filename,
            "total_faces": len(detections),
            "detections": detections,
            "execution_time_ms": exec_time_ms
        }

        return result_payload, annotated_bgr

    def process_video_bytes(
        self,
        video_bytes: bytes,
        filename: str = "video.mp4",
        frame_stride: int = DEFAULT_VIDEO_FRAME_STRIDE,
        conf_thresh: float = None,
        emotion_conf_thresh: float = None,
        margin: float = None
    ) -> Tuple[Dict[str, Any], bytes]:
        start_time = time.time()
        if conf_thresh is None:
            conf_thresh = FACE_CONF_THRESH
        if emotion_conf_thresh is None:
            emotion_conf_thresh = EMOTION_CONF_THRESH
        if frame_stride is None or frame_stride < 1:
            frame_stride = DEFAULT_VIDEO_FRAME_STRIDE

        from .detector import CentroidFaceTracker
        local_tracker = CentroidFaceTracker()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as in_file:
            in_file.write(video_bytes)
            in_path = in_file.name

        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = out_file.name
        out_file.close()

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            os.remove(in_path)
            raise ValueError("Could not open or decode uploaded video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_records = []
        annotated_frames_rgb = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                faces = self.detector.detect(frame, conf_thresh=conf_thresh, margin=margin)
                faces = local_tracker.update(faces)
                annotated_frame = frame.copy()
                frame_faces = []

                for idx, face in enumerate(faces):
                    if face["det_conf"] < conf_thresh:
                        continue
                    x1, y1, x2, y2 = face["bbox"]
                    crop = frame[y1:y2, x1:x2]
                    emo_res = self.classifier.classify(crop)

                    if emo_res["cls_conf"] < emotion_conf_thresh:
                        continue

                    track_id = face.get("track_id", idx + 1)

                    rec = {
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "face_id": idx,
                        "track_id": track_id,
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "face_confidence": round(face["det_conf"], 4),
                        "emotion_label": emo_res["label"],
                        "emotion_confidence": round(emo_res["cls_conf"], 4)
                    }
                    frame_faces.append(rec)

                    # Dynamic scaling for text & bounding box
                    thickness = max(2, int(min(w, h) / 300))
                    font_scale = max(0.4, min(w, h) / 800)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
                    text = f"#{track_id} {emo_res['label']} (Face:{face['det_conf']*100:.0f}%, Emo:{emo_res['cls_conf']*100:.0f}%)"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                    cv2.rectangle(annotated_frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, text, (x1 + 2, max(th, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_frames_rgb.append(annotated_rgb)

                frame_records.append({
                    "frame": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "total_faces": len(frame_faces),
                    "detections": frame_faces
                })

            frame_idx += 1

        cap.release()
        os.remove(in_path)

        out_bytes = b""
        if annotated_frames_rgb:
            try:
                import imageio
                out_fps = max(1.0, float(fps) / frame_stride)
                writer = imageio.get_writer(out_path, fps=out_fps, codec="libx264", macro_block_size=1)
                for f_rgb in annotated_frames_rgb:
                    writer.append_data(f_rgb)
                writer.close()

                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        out_bytes = f.read()
                    os.remove(out_path)
            except Exception as e:
                # Fallback to OpenCV VideoWriter if imageio fails
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, max(1.0, fps / frame_stride), (w, h))
                for f_rgb in annotated_frames_rgb:
                    f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
                    writer.write(f_bgr)
                writer.release()
                if os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        out_bytes = f.read()
                    os.remove(out_path)

        exec_time_ms = round((time.time() - start_time) * 1000, 2)
        annotated_video_b64 = base64.b64encode(out_bytes).decode('utf-8') if out_bytes else ""

        payload = {
            "status": "success",
            "filename": filename,
            "total_frames_processed": len(frame_records),
            "frame_stride": frame_stride,
            "frame_timeline": frame_records,
            "execution_time_ms": exec_time_ms,
            "annotated_video_b64": annotated_video_b64
        }

        return payload, out_bytes
