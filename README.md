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
                               | HTTP POST /predict
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
| (YOLOv8 Face Engine)  |----Crop---->| (7-Class Classifier)  |
+-----------------------+             +-----------------------+
```

* **2-Stage ML Pipeline:**
  1. **Face Detector (`yolov8n-face`):** Detects face bounding boxes and applies a dynamic crop expansion margin.
  2. **Emotion Classifier:** Classifies cropped facial regions into 7 emotion classes (*Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*).
* **Multi-Modal UI (Streamlit):**
  * **Single Image Analysis:** Visualizes bounding boxes, emotion probability distributions, and execution latency.
  * **Batch Image Processing:** Ingests multiple images and exports structured CSV reports (`emotion_batch_predictions.csv`).
  * **Live Camera Snapshot:** Directly processes browser webcam snapshots in real time.
* **Production Ready:** Fully containerized with Docker and Docker Compose, supported by a 100% green Pytest unit & integration test suite.

---

## 🚀 How to Run the Application

### Method 1: Using Docker Compose (Recommended)

> **Yes!** `docker-compose up --build` automatically builds and runs **both** the backend and frontend services simultaneously in isolated containers.

```bash
# Build and start both Backend and Frontend containers
docker-compose up --build
```

#### Access Points:
* **Frontend Web App (Streamlit):** [http://localhost:8501](http://localhost:8501)
* **Backend API (FastAPI):** [http://localhost:8080](http://localhost:8080)
* **Interactive API Documentation (Swagger):** [http://localhost:8080/docs](http://localhost:8080/docs)

To stop the containers:
```bash
docker-compose down
```

---

### Method 2: Running Locally with Python

#### 1. Setup Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install backend & frontend requirements
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

#### 2. Model Weights (Optional)
Place your trained model files in `backend/models/`:
* `yolov8n-face.pt` (Face Detector)
* `emotion_model.pt` (Emotion Classifier)

*Note: If model files are not present, the system automatically runs in **Mock Mode** for demonstration and local testing.*

#### 3. Start Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8080
```

#### 4. Start Frontend Interface (in a new terminal)
```bash
streamlit run frontend/app.py
```

---

## 🧪 Running the Test Suite

Run the automated Pytest suite to verify image decoding, preprocessing, 2-stage inference pipeline, and API endpoints:

```bash
python -m pytest backend/tests/
```

---

## 📡 API Endpoints Overview

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API welcome message & link to docs |
| `/health` | `GET` | Health check endpoint returning loaded model status |
| `/predict` | `POST` | Accepts an image file, returns JSON with bounding boxes & emotion probabilities |
| `/predict-annotated` | `POST` | Accepts an image file, returns a JPEG byte stream with rendered detection overlays |

---

## 📂 Project Structure

```
Facial_Expression_Recognition/
├── backend/
│   ├── app/
│   │   ├── config.py         # Global settings & threshold defaults
│   │   ├── schemas.py        # Pydantic request/response data models
│   │   ├── preprocess.py     # Image decoding, validation & bbox expansion
│   │   ├── detector.py       # YOLOv8 face detector wrapper
│   │   ├── classifier.py     # Emotion classification engine
│   │   ├── pipeline.py       # 2-Stage inference controller
│   │   └── main.py           # FastAPI app entrypoint
│   ├── models/               # Model weights (.pt files)
│   ├── tests/                # Pytest unit & integration tests
│   └── Dockerfile            # Backend Docker image config
├── frontend/
│   ├── components/
│   │   └── visualizer.py     # Bounding box & emotion tag renderer
│   ├── app.py                # Streamlit interactive application
│   └── Dockerfile            # Frontend Docker image config
├── docker-compose.yml        # Orchestrates Backend & Frontend containers
└── README.md
```
