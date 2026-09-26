import cv2
import numpy as np
from PIL import Image

EMOTION_COLOR_MAP = {
    "Angry": (239, 68, 68),     # Red
    "Disgust": (168, 85, 247),  # Purple
    "Fear": (249, 115, 22),    # Orange
    "Happy": (34, 197, 94),     # Green
    "Sad": (59, 130, 246),      # Blue
    "Surprise": (236, 72, 153), # Pink
    "Neutral": (107, 114, 128)  # Gray
}

def render_predictions(image_bytes: bytes, detections: list) -> Image.Image:
    """Draws bounding boxes and custom colored emotion tags onto image."""
    img_pil = Image.open(image_bytes).convert("RGB")
    img_np = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        label = det["emotion_label"]
        conf = det["emotion_confidence"]

        # Color in BGR
        rgb_color = EMOTION_COLOR_MAP.get(label, (50, 150, 250))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])

        # Rectangle
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), bgr_color, 3)

        # Label pill background
        tag_text = f"{label} ({conf*100:.1f}%)"
        (tw, th), baseline = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img_bgr, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), bgr_color, -1)
        cv2.putText(img_bgr, tag_text, (x1 + 4, max(th, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
