import os
import io
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from components.visualizer import render_predictions

# Page Configuration
st.set_page_config(
    page_title="Face Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("⚙️ Model Parameters")
BACKEND_URL = st.sidebar.text_input("Backend API URL", value=os.getenv("BACKEND_URL", "http://localhost:8080"))

st.sidebar.markdown("---")
FACE_CONF_THRESH = st.sidebar.slider("Face Detection Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)
FACE_CROP_MARGIN = st.sidebar.slider("Face Crop Expansion Margin", min_value=0.00, max_value=0.30, value=0.15, step=0.05)

st.sidebar.markdown("---")
# Health Status Check
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        health_data = health_resp.json()
        st.sidebar.success("🟢 API Status: Connected")
        st.sidebar.caption(f"Face Model Loaded: {'Yes' if health_data['face_model_loaded'] else 'Mock Mode'}")
        st.sidebar.caption(f"Emotion Model Loaded: {'Yes' if health_data['emotion_model_loaded'] else 'Mock Mode'}")
    else:
        st.sidebar.warning("🟡 API Status: Degraded")
except Exception:
    st.sidebar.error("🔴 API Status: Disconnected")
    st.sidebar.caption("Make sure FastAPI backend is running.")

# Title Header
st.markdown('<div class="main-header">🎭 Real-time Face Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload face photos or snapshots to analyze emotion predictions with confidence scores.</div>', unsafe_allow_html=True)

# Application Tabs
tab1, tab2, tab3 = st.tabs(["📸 Single Image Analysis", "📁 Batch Processing", "🎥 Live Camera Snapshot"])

def call_api(image_bytes: bytes, filename: str):
    params = {
        "conf_thresh": FACE_CONF_THRESH,
        "margin": FACE_CROP_MARGIN
    }
    files = {"file": (filename, image_bytes, "image/jpeg")}
    response = requests.post(f"{BACKEND_URL}/predict", params=params, files=files, timeout=10)
    return response

with tab1:
    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Upload Face Image")
        uploaded_file = st.file_uploader("Choose a JPEG / PNG image", type=["jpg", "jpeg", "png", "webp"], key="single_upload")

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption="Original Image", use_container_width=True)

    with col_output:
        st.subheader("Prediction Results")
        if uploaded_file is not None:
            with st.spinner("Analyzing image..."):
                try:
                    resp = call_api(image_bytes, uploaded_file.name)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Metrics row
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{data["total_faces"]}</div><div class="metric-label">Detected Faces</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{data["execution_time_ms"]} ms</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                        with m3:
                            top_emo = data["detections"][0]["emotion_label"] if data["total_faces"] > 0 else "N/A"
                            st.markdown(f'<div class="metric-card"><div class="metric-value">{top_emo}</div><div class="metric-label">Primary Emotion</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        # Render annotated image
                        if data["total_faces"] > 0:
                            annotated_pil = render_predictions(image_bytes, data["detections"])
                            st.image(annotated_pil, caption="Annotated Detections", use_container_width=True)

                            # Detailed breakdown
                            st.markdown("#### Detailed Face Detections")
                            for det in data["detections"]:
                                with st.expander(f"Face #{det['face_id']+1} — {det['emotion_label']} ({det['emotion_confidence']*100:.1f}%)"):
                                    st.write(f"**Bounding Box:** `{det['bbox']}`")
                                    st.write(f"**Detection Confidence:** `{det['face_confidence']*100:.1f}%`")
                                    if det.get("all_scores"):
                                        st.write("**Emotion Class Distribution:**")
                                        df_scores = pd.DataFrame(list(det["all_scores"].items()), columns=["Emotion", "Probability"])
                                        st.bar_chart(df_scores.set_index("Emotion"))
                        else:
                            st.info("No faces detected in this image. Try adjusting the face detection threshold in the sidebar.")

                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Connection failed: Could not reach backend at `{BACKEND_URL}`. Details: {e}")
        else:
            st.info("Please upload an image to begin emotion classification.")

with tab2:
    st.subheader("Batch Image Processing")
    uploaded_files = st.file_uploader("Upload multiple face images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Run Batch Inference", type="primary"):
            results = []
            progress_bar = st.progress(0)

            for idx, file in enumerate(uploaded_files):
                img_bytes = file.getvalue()
                try:
                    resp = call_api(img_bytes, file.name)
                    if resp.status_code == 200:
                        payload = resp.json()
                        for det in payload["detections"]:
                            results.append({
                                "Filename": file.name,
                                "Face ID": det["face_id"],
                                "Emotion": det["emotion_label"],
                                "Emotion Conf": f"{det['emotion_confidence']*100:.1f}%",
                                "Face Conf": f"{det['face_confidence']*100:.1f}%",
                                "Bounding Box": str(det["bbox"])
                            })
                        if payload["total_faces"] == 0:
                            results.append({
                                "Filename": file.name,
                                "Face ID": "-",
                                "Emotion": "NO_FACE_DETECTED",
                                "Emotion Conf": "-",
                                "Face Conf": "-",
                                "Bounding Box": "-"
                            })
                except Exception as e:
                    results.append({"Filename": file.name, "Emotion": f"ERROR: {e}"})

                progress_bar.progress((idx + 1) / len(uploaded_files))

            df_batch = pd.DataFrame(results)
            st.markdown("### Batch Results Summary")
            st.dataframe(df_batch, use_container_width=True)

            # Download CSV
            csv_data = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", data=csv_data, file_name="emotion_batch_predictions.csv", mime="text/csv")

with tab3:
    st.subheader("Live Camera Snapshot")
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        cam_bytes = camera_photo.getvalue()
        with st.spinner("Processing camera snapshot..."):
            try:
                resp = call_api(cam_bytes, "webcam_snapshot.jpg")
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Detected {data['total_faces']} face(s) in {data['execution_time_ms']} ms")

                    if data["total_faces"] > 0:
                        annotated_cam = render_predictions(cam_bytes, data["detections"])
                        st.image(annotated_cam, caption="Live Detection Result", use_container_width=True)
                    else:
                        st.warning("No face detected in camera photo.")
            except Exception as e:
                st.error(f"Failed to process snapshot: {e}")
