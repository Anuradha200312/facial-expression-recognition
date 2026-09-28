from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class FaceDetectionResult(BaseModel):
    face_id: int
    track_id: Optional[int] = None
    bbox: BoundingBox
    face_confidence: float = Field(..., ge=0.0, le=1.0)
    emotion_label: str
    emotion_confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: Optional[Dict[str, float]] = None

class PredictionResponse(BaseModel):
    status: str
    filename: str
    total_faces: int
    detections: List[FaceDetectionResult]
    execution_time_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    service: str
    face_model_loaded: bool
    emotion_model_loaded: bool
