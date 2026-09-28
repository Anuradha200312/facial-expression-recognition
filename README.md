# 🎭 Face Emotion Recognition System

An end-to-end 2-stage Computer Vision system featuring a **FastAPI backend API**, a **PyTorch / YOLOv8 ML engine**, and an interactive **Streamlit frontend dashboard**.

---

## 🌟 Architecture & Features

```
                   +------------------------+
                   |   User Interface       |
                   |   (Streamlit Frontend) |
                   +-----------+------------+
                               |
                               | HTTP POST /predict, /predict-zip, /predict-video
                               v
                   +------------------------+
                   |     FastAPI Backend    |
                   +-----------+------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+             +-----------------------+
| Stage 1: Face Detector|             |Stage 2: Emotion Model |
| (face_model.pt)       |----Crop---->| (emotions_model.pt)   |
+-----------------------+             +-----------------------+
```

### 🧠 Model Weights Convention (`backend/models/`)
1. **`face_model.pt`**: Face detection model (YOLOv8 face detector).
2. **`emotions_model.pt`**: Emotion classification model (7-class emotion model).
* **Smart Auto-Discovery Fallback:** If default filenames are not present, the system automatically scans `.pt` files in `backend/models/` and inspects YOLO task/class metadata to assign detector and classifier roles automatically.

---

### ✨ Features Overview

* **📸 Single Image Analysis:** Instant face detection, color-coded emotion pills, probability distribution charts, and latency metrics.
* **📦 ZIP Archive Batch Extraction:** Upload `.zip` files containing images or nested folders. Recursively extracts images, validates formats, executes batch inference, and exports structured CSV reports (`zip_batch_predictions.csv`).
* **🎞️ Video Stream Processing:** Ingests video files (`.mp4`, `.avi`, `.mov`, `.mkv`), applies configurable frame striding (`VIDEO_FRAME_STRIDE = 5`) to ensure responsive processing without GPU memory exhaustion, and exports frame timeline CSVs.
* **🎥 Live Camera Stream with Start/Stop Controls:** Interactive live webcam feed toggle (**"Start Live Stream"** / **"Stop Stream"**) with real-time bounding box overlays and emotion prediction metrics.

---

## 🚀 How to Run the Application

### Method 1: Using Docker Compose (Recommended)

```bash
# Build and start both Backend and Frontend containers
docker-compose up --build
```

#### Access Points:
* **Frontend Web App (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **Backend API (FastAPI):** [http://localhost:8080](http://localhost:8080)
* **Interactive API Documentation (Swagger):** [http://localhost:8080/docs](http://localhost:8080/docs)

---

### Method 2: Running Locally with Python

#### 1. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate    # Linux/macOS

pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

#### 2. Start Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8080
```

#### 3. Start Frontend Dashboard
```bash
streamlit run frontend/app.py
```

---

## 🧪 Running the Test Suite

Run the automated Pytest suite (13 unit & integration tests):

```bash
python -m pytest backend/tests/
```

---

## 📡 API Endpoints Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API root status |
| `/health` | `GET` | Health check returning loaded status for `face_model.pt` & `emotions_model.pt` |
| `/predict` | `POST` | Accepts single image file, returns JSON bounding boxes & emotion probabilities |
| `/predict-annotated` | `POST` | Accepts single image file, returns JPEG byte stream with rendered annotations |
| `/predict-zip` | `POST` | Accepts `.zip` archive, extracts valid images, executes batch prediction, returns JSON & rejections log |
| `/predict-video` | `POST` | Accepts video file, processes strided frames, returns frame-by-frame emotion timeline |
