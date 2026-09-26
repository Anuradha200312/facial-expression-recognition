import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionResponse, HealthCheckResponse
from .pipeline import Pipeline
from .preprocess import ImageValidationError

app = FastAPI(
    title="Face Emotion Recognition API",
    description="FastAPI service for 2-stage face detection and emotion classification.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = Pipeline()

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Face Emotion Recognition API is running.",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["General"])
def health_check():
    return {
        "status": "ok",
        "service": "Face Emotion Recognition API",
        "face_model_loaded": pipeline.detector.is_loaded,
        "emotion_model_loaded": pipeline.classifier.is_loaded
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0, description="Minimum face detection confidence threshold"),
    margin: float = Query(None, ge=0.0, le=0.5, description="Face crop expansion margin ratio")
):
    try:
        contents = await file.read()
        payload, _ = pipeline.process_bytes(contents, filename=file.filename or "uploaded.jpg", conf_thresh=conf_thresh, margin=margin)
        return payload
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.post("/predict-annotated", tags=["Prediction"])
async def predict_annotated(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5)
):
    try:
        contents = await file.read()
        _, annotated_bgr = pipeline.process_bytes(contents, filename=file.filename or "uploaded.jpg", conf_thresh=conf_thresh, margin=margin)
        success, encoded_image = cv2.imencode(".jpg", annotated_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")
