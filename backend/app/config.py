import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = os.getenv("MODEL_DIR", str(BASE_DIR / "models"))

DEFAULT_FACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_model.pt")
DEFAULT_EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotions_model.pt")

def resolve_model_paths():
    """Resolves face and emotion model paths using explicit env variables, default filenames, or auto-discovery."""
    face_path = os.getenv("FACE_MODEL_PATH", DEFAULT_FACE_MODEL_PATH)
    emotion_path = os.getenv("EMOTION_MODEL_PATH", DEFAULT_EMOTION_MODEL_PATH)

    # Auto-discovery fallback if default files are not found
    if not os.path.exists(face_path) or not os.path.exists(emotion_path):
        if os.path.exists(MODEL_DIR):
            pt_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
            for pt in pt_files:
                fname = os.path.basename(pt).lower()
                if not os.path.exists(face_path) and ("face" in fname or "yolo" in fname):
                    face_path = pt
                elif not os.path.exists(emotion_path) and ("emotion" in fname or "cls" in fname):
                    emotion_path = pt

    return face_path, emotion_path

FACE_MODEL_PATH, EMOTION_MODEL_PATH = resolve_model_paths()

FACE_CONF_THRESH = float(os.getenv("FACE_CONF_THRESH", "0.65"))
EMOTION_CONF_THRESH = float(os.getenv("EMOTION_CONF_THRESH", "0.65"))
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.15"))

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_RESOLUTION = 20
DEFAULT_VIDEO_FRAME_STRIDE = 3
