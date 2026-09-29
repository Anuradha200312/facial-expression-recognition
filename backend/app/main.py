import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictionResponse, HealthCheckResponse
from .pipeline import Pipeline
from .preprocess import ImageValidationError, extract_images_from_zip
from .config import DEFAULT_VIDEO_FRAME_STRIDE, FACE_CONF_THRESH, EMOTION_CONF_THRESH

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
def predict(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0, description="Minimum face detection confidence threshold"),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0, description="Minimum emotion confidence threshold"),
    margin: float = Query(None, ge=0.0, le=0.5, description="Face crop expansion margin ratio"),
    is_stream: bool = Query(False, description="Whether input is live camera frame for tracking")
):
    try:
        contents = file.file.read()
        payload, _ = pipeline.process_bytes(
            contents,
            filename=file.filename or "uploaded.jpg",
            conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin,
            is_stream=is_stream
        )
        return payload
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.post("/predict-annotated", tags=["Prediction"])
def predict_annotated(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5),
    is_stream: bool = Query(False)
):
    try:
        contents = file.file.read()
        _, annotated_bgr = pipeline.process_bytes(
            contents,
            filename=file.filename or "uploaded.jpg",
            conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin,
            is_stream=is_stream
        )
        success, encoded_image = cv2.imencode(".jpg", annotated_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
        return Response(content=encoded_image.tobytes(), media_type="image/jpeg")
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")

@app.post("/predict-zip", tags=["Prediction"])
def predict_zip(
    file: UploadFile = File(...),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5)
):
    try:
        contents = file.file.read()
        valid_images, rejections = extract_images_from_zip(contents)

        results = []
        for filename, img_bytes in valid_images:
            payload, _ = pipeline.process_bytes(
                img_bytes,
                filename=filename,
                conf_thresh=conf_thresh,
                emotion_conf_thresh=emotion_conf_thresh,
                margin=margin
            )
            results.append(payload)

        return {
            "status": "success",
            "archive_filename": file.filename,
            "total_extracted_images": len(valid_images),
            "total_rejected_files": len(rejections),
            "results": results,
            "rejections": rejections
        }
    except ImageValidationError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process zip archive: {str(e)}")

@app.post("/predict-video", tags=["Prediction"])
def predict_video(
    file: UploadFile = File(...),
    frame_stride: int = Query(DEFAULT_VIDEO_FRAME_STRIDE, ge=1, le=30),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5)
):
    try:
        contents = file.file.read()
        payload, _ = pipeline.process_video_bytes(
            contents,
            filename=file.filename or "video.mp4",
            frame_stride=frame_stride,
            conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin
        )
        return payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.post("/predict-video-file", tags=["Prediction"])
def predict_video_file(
    file: UploadFile = File(...),
    frame_stride: int = Query(DEFAULT_VIDEO_FRAME_STRIDE, ge=1, le=30),
    conf_thresh: float = Query(None, ge=0.0, le=1.0),
    emotion_conf_thresh: float = Query(None, ge=0.0, le=1.0),
    margin: float = Query(None, ge=0.0, le=0.5)
):
    try:
        contents = file.file.read()
        _, out_bytes = pipeline.process_video_bytes(
            contents,
            filename=file.filename or "video.mp4",
            frame_stride=frame_stride,
            conf_thresh=conf_thresh,
            emotion_conf_thresh=emotion_conf_thresh,
            margin=margin
        )
        return Response(content=out_bytes, media_type="video/mp4")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video file processing failed: {str(e)}")

