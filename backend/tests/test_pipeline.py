import numpy as np
from app.pipeline import Pipeline

def test_pipeline_process_bytes(sample_valid_image_bytes):
    pipeline = Pipeline()
    payload, annotated_bgr = pipeline.process_bytes(sample_valid_image_bytes, filename="test.jpg")

    assert payload["status"] == "success"
    assert payload["filename"] == "test.jpg"
    assert isinstance(payload["total_faces"], int)
    assert isinstance(payload["detections"], list)
    assert payload["execution_time_ms"] > 0
    assert isinstance(annotated_bgr, np.ndarray)
    assert annotated_bgr.shape == (100, 100, 3)

def test_pipeline_detections_format(sample_valid_image_bytes):
    pipeline = Pipeline()
    payload, _ = pipeline.process_bytes(sample_valid_image_bytes)

    if payload["total_faces"] > 0:
        det = payload["detections"][0]
        assert "face_id" in det
        assert "bbox" in det
        assert "face_confidence" in det
        assert "emotion_label" in det
        assert "emotion_confidence" in det
        assert 0.0 <= det["face_confidence"] <= 1.0
        assert 0.0 <= det["emotion_confidence"] <= 1.0
