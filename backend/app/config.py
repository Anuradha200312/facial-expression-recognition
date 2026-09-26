import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))

FACE_MODEL_PATH = os.getenv("FACE_MODEL_PATH", os.path.join(MODEL_DIR, "yolov8n-face.pt"))
EMOTION_MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", os.path.join(MODEL_DIR, "emotion_model.pt"))

FACE_CONF_THRESH = float(os.getenv("FACE_CONF_THRESH", "0.40"))
EMOTION_CONF_THRESH = float(os.getenv("EMOTION_CONF_THRESH", "0.00"))
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.15"))

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RESOLUTION = 20
