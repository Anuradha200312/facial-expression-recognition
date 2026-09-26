import io
from PIL import Image, UnidentifiedImageError
import numpy as np
import cv2
from .config import ALLOWED_EXTENSIONS, MIN_RESOLUTION

class ImageValidationError(Exception):
    pass

def validate_and_decode_image(image_bytes: bytes) -> np.ndarray:
    """Decodes image bytes to BGR numpy array after validation."""
    if not image_bytes or len(image_bytes) == 0:
        raise ImageValidationError("Uploaded file is empty.")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        raise ImageValidationError(f"Invalid or corrupted image file: {str(e)}")

    # Re-open after verify() closes the handle
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        raise ImageValidationError(f"Image resolution too small ({w}x{h}). Minimum required is {MIN_RESOLUTION}x{MIN_RESOLUTION}.")

    # Convert PIL Image to OpenCV BGR format
    img_rgb = img.convert("RGB")
    img_np = np.array(img_rgb)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    return img_bgr

def expand_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, margin: float = 0.15):
    """Expands bounding box by margin ratio while keeping coordinates within image boundaries."""
    bw, bh = x2 - x1, y2 - y1
    ex_x1 = max(0, int(x1 - bw * margin))
    ex_y1 = max(0, int(y1 - bh * margin))
    ex_x2 = min(img_w, int(x2 + bw * margin))
    ex_y2 = min(img_h, int(y2 + bh * margin))
    return ex_x1, ex_y1, ex_x2, ex_y2
