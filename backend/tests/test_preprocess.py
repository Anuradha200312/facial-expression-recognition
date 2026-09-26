import pytest
import numpy as np
from app.preprocess import validate_and_decode_image, expand_bbox, ImageValidationError

def test_validate_and_decode_valid_image(sample_valid_image_bytes):
    img_bgr = validate_and_decode_image(sample_valid_image_bytes)
    assert isinstance(img_bgr, np.ndarray)
    assert img_bgr.shape == (100, 100, 3)

def test_validate_and_decode_empty_bytes():
    with pytest.raises(ImageValidationError, match="Uploaded file is empty"):
        validate_and_decode_image(b"")

def test_validate_and_decode_corrupt_bytes(sample_corrupt_bytes):
    with pytest.raises(ImageValidationError, match="Invalid or corrupted image file"):
        validate_and_decode_image(sample_corrupt_bytes)

def test_validate_and_decode_too_small_resolution(sample_small_image_bytes):
    with pytest.raises(ImageValidationError, match="Image resolution too small"):
        validate_and_decode_image(sample_small_image_bytes)

def test_expand_bbox_boundary():
    x1, y1, x2, y2 = expand_bbox(20, 20, 80, 80, img_w=100, img_h=100, margin=0.10)
    # bw=60, bh=60 -> margin=6 -> x1=14, y1=14, x2=86, y2=86
    assert x1 == 14
    assert y1 == 14
    assert x2 == 86
    assert y2 == 86

    # Test out of bounds clamping
    ex1, ey1, ex2, ey2 = expand_bbox(0, 0, 50, 50, img_w=100, img_h=100, margin=0.50)
    assert ex1 == 0
    assert ey1 == 0
    assert ex2 == 75
    assert ey2 == 75
