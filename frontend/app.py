import os
import io
import time
import queue
import base64
import zipfile
import requests
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
from components.visualizer import (
    render_predictions,
    render_predictions_bytes,
    create_video_from_frames,
    EMOTION_COLOR_MAP
)

# Fixed System Thresholds & Configurations (Code-Level Controlled)
FACE_CONF_THRESH = 0.65
EMOTION_CONF_THRESH = 0.65
VIDEO_FRAME_STRIDE = 3
FACE_CROP_MARGIN = 0.15

# Page Setup
st.set_page_config(
    page_title="AI Facial Emotion Intelligence",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Glassmorphism & Sleek Design System
st.markdown("""
<style>
    /* Global Container Theme */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Sleek Header & Titles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }

    /* Metric Panel Styling */
    .metric-panel {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Threshold Badge Info Cards */
    .thresh-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .thresh-label {
        font-size: 0.8rem;
        color: #94A3B8;
    }
    .thresh-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60A5FA;
    }

    /* Status Pill Badges */
    .badge-online {
        background-color: rgba(34, 197, 94, 0.15);
        color: #4ADE80;
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-offline {
        background-color: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Tab Headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        color: #94A3B8;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Control & Configuration Hub
st.sidebar.title("🎛️ Pipeline Settings")
BACKEND_URL = st.sidebar.text_input("Backend API Endpoint", value=os.getenv("BACKEND_URL", "http://localhost:8080"))

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Fixed Code Configurations")
st.sidebar.markdown(f"""
<div class="thresh-box">
    <div class="thresh-label">Face Detection Confidence</div>
    <div class="thresh-val">65% ({FACE_CONF_THRESH})</div>
</div>
<div class="thresh-box">
    <div class="thresh-label">Emotion Classification Confidence</div>
    <div class="thresh-val">65% ({EMOTION_CONF_THRESH})</div>
</div>
<div class="thresh-box">
    <div class="thresh-label">Video / Live Frame Subsampling</div>
    <div class="thresh-val">Every 3rd Frame (N=3)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Diagnostic Status")

# API Health Diagnostic
try:
    health_resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_resp.status_code == 200:
        h_data = health_resp.json()
        st.sidebar.markdown('<span class="badge-online">🟢 API Server Connected</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"**Face Detector:** `face_model.pt` ({'Loaded' if h_data['face_model_loaded'] else 'Mock Mode'})")
        st.sidebar.caption(f"**Emotion Classifier:** `emotions_model.pt` ({'Loaded' if h_data['emotion_model_loaded'] else 'Mock Mode'})")
    else:
        st.sidebar.markdown('<span class="badge-offline">🟡 API Server Degraded</span>', unsafe_allow_html=True)
except Exception:
    st.sidebar.markdown('<span class="badge-offline">🔴 API Server Disconnected</span>', unsafe_allow_html=True)
    st.sidebar.caption("Run `uvicorn backend.app.main:app --port 8080` to launch backend.")

# Header Title
st.markdown('<div class="main-header">🎭 Real-Time Face Emotion Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">2-Stage Deep Learning Pipeline for Face Detection, Persistent Face Tracking, and Multi-Class Emotion Recognition.</div>', unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Single Image",
    "📦 Batch & Grid Processing",
    "🎞️ Video Analysis",
    "🎥 Live Stream Camera"
])

# Helper function to call prediction API
def call_predict_api(image_bytes: bytes, filename: str = "upload.jpg", is_stream: bool = False):
    params = {
        "conf_thresh": FACE_CONF_THRESH,
        "emotion_conf_thresh": EMOTION_CONF_THRESH,
        "margin": FACE_CROP_MARGIN,
        "is_stream": is_stream
    }
    files = {"file": (filename, image_bytes, "image/jpeg")}
    return requests.post(f"{BACKEND_URL}/predict", params=params, files=files, timeout=15)

# -------------------------------------------------------------------
# TAB 1: Single Image Analysis
# -------------------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_image = st.file_uploader("Upload a face photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"], key="single_img_upload")
        if uploaded_image:
            image_bytes = uploaded_image.getvalue()
            st.image(image_bytes, caption="Source Upload", use_container_width=True)

    with col2:
        st.markdown("### 📊 Detection & Emotion Output")
        if uploaded_image:
            with st.spinner("Analyzing image..."):
                try:
                    resp = call_predict_api(image_bytes, uploaded_image.name)
                    if resp.status_code == 200:
                        data = resp.json()

                        # Metrics Display Cards
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.markdown(f'<div class="metric-panel"><div class="metric-value">{data["total_faces"]}</div><div class="metric-label">Detected Faces</div></div>', unsafe_allow_html=True)
                        with m2:
                            st.markdown(f'<div class="metric-panel"><div class="metric-value">{data["execution_time_ms"]} ms</div><div class="metric-label">Latency</div></div>', unsafe_allow_html=True)
                        with m3:
                            top_emo = data["detections"][0]["emotion_label"] if data["total_faces"] > 0 else "None"
                            st.markdown(f'<div class="metric-panel"><div class="metric-value">{top_emo}</div><div class="metric-label">Primary Emotion</div></div>', unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)

                        if data["total_faces"] > 0:
                            annotated_pil = render_predictions(image_bytes, data["detections"])
                            st.image(annotated_pil, caption="Pipeline Annotations (Bounding Box, Track ID, Face Conf ≥ 65%, Emotion Conf ≥ 65%)", use_container_width=True)

                            # Individual Download Button for single image
                            img_download_bytes = render_predictions_bytes(image_bytes, data["detections"], format="JPEG")
                            st.download_button(
                                label="📥 Download Annotated Image (.jpg)",
                                data=img_download_bytes,
                                file_name=f"annotated_{uploaded_image.name}",
                                mime="image/jpeg",
                                type="primary",
                                use_container_width=True
                            )

                            st.markdown("#### 🎯 Per-Face Emotion Analysis")
                            for det in data["detections"]:
                                track_str = f"Face #{det.get('track_id', det['face_id']+1)}"
                                with st.expander(f"{track_str} — {det['emotion_label']} (Face: {det['face_confidence']*100:.1f}%, Emo: {det['emotion_confidence']*100:.1f}%)"):
                                    st.write(f"**Bounding Box:** `{det['bbox']}`")
                                    st.write(f"**Face Detection Confidence:** `{det['face_confidence']*100:.1f}%`")
                                    st.write(f"**Emotion Classification Confidence:** `{det['emotion_confidence']*100:.1f}%`")
                                    if det.get("all_scores"):
                                        df_scores = pd.DataFrame(list(det["all_scores"].items()), columns=["Emotion", "Probability"])
                                        st.bar_chart(df_scores.set_index("Emotion"))
                        else:
                            annotated_pil = render_predictions(image_bytes, [])
                            st.image(annotated_pil, caption="No faces detected above confidence thresholds.", use_container_width=True)
                            st.warning("No faces detected above the 65% detection & emotion confidence thresholds.")
                    else:
                        st.error(f"API Error ({resp.status_code}): {resp.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
        else:
            st.info("Upload an image to trigger face detection and emotion classification.")

# -------------------------------------------------------------------
# TAB 2: Batch & Grid Processing (4 Images Per Row)
# -------------------------------------------------------------------
with tab2:
    st.markdown("### 📦 Batch Upload & 4-Column Output Grid")
    st.markdown("Upload multiple images or a **`.zip` archive**. Outputs are arranged in a **grid of 4 images per row**, each with an individual annotated image download button.")

    batch_files = st.file_uploader(
        "Upload ZIP archive or multiple face images (JPG, PNG, WEBP, ZIP)",
        type=["zip", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_files_upload"
    )

    if batch_files:
        if st.button("🚀 Process Batch Images & ZIP Archives", type="primary"):
            annotated_results = [] # list of dicts: {"filename", "image_bytes", "detections", "total_faces"}
            csv_records = []

            with st.spinner("Processing batch images and zip contents..."):
                for b_file in batch_files:
                    fn = b_file.name
                    b_bytes = b_file.getvalue()

                    if fn.lower().endswith(".zip"):
                        # Extract and process each image inside zip archive
                        try:
                            with zipfile.ZipFile(io.BytesIO(b_bytes), "r") as z:
                                for member in z.infolist():
                                    if member.is_dir():
                                        continue
                                    z_fn = member.filename
                                    # Skip hidden/OS metadata files
                                    if z_fn.startswith("__MACOSX") or os.path.basename(z_fn).startswith("."):
                                        continue
                                    ext = os.path.splitext(z_fn)[1].lower()
                                    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                                        img_bytes = z.read(member)
                                        clean_fn = os.path.basename(z_fn)
                                        try:
                                            resp = call_predict_api(img_bytes, clean_fn)
                                            if resp.status_code == 200:
                                                data = resp.json()
                                                dets = data.get("detections", [])
                                                annotated_results.append({
                                                    "filename": z_fn,
                                                    "image_bytes": img_bytes,
                                                    "detections": dets,
                                                    "total_faces": data["total_faces"]
                                                })
                                                if not dets:
                                                    csv_records.append({
                                                        "Archive": fn, "Filename": z_fn, "Total Faces": 0,
                                                        "Primary Emotion": "NO_FACE_DETECTED", "Emotion Conf": "-",
                                                        "Face Conf": "-", "Bounding Box": "-"
                                                    })
                                                else:
                                                    for det in dets:
                                                        csv_records.append({
                                                            "Archive": fn, "Filename": z_fn,
                                                            "Track ID": det.get("track_id", det["face_id"] + 1),
                                                            "Total Faces": data["total_faces"],
                                                            "Emotion": det["emotion_label"],
                                                            "Emotion Conf": f"{det['emotion_confidence']*100:.1f}%",
                                                            "Face Conf": f"{det['face_confidence']*100:.1f}%",
                                                            "Bounding Box": str(det["bbox"])
                                                        })
                                        except Exception as e:
                                            st.error(f"Failed to process zip image '{z_fn}': {e}")
                        except zipfile.BadZipFile:
                            st.error(f"Corrupted or invalid zip file: {fn}")
                    else:
                        # Individual image uploaded in batch
                        try:
                            resp = call_predict_api(b_bytes, fn)
                            if resp.status_code == 200:
                                data = resp.json()
                                dets = data.get("detections", [])
                                annotated_results.append({
                                    "filename": fn,
                                    "image_bytes": b_bytes,
                                    "detections": dets,
                                    "total_faces": data["total_faces"]
                                })

                                if not dets:
                                    csv_records.append({
                                        "Archive": "Direct Upload", "Filename": fn, "Total Faces": 0,
                                        "Primary Emotion": "NO_FACE_DETECTED", "Emotion Conf": "-",
                                        "Face Conf": "-", "Bounding Box": "-"
                                    })
                                else:
                                    for det in dets:
                                        csv_records.append({
                                            "Archive": "Direct Upload", "Filename": fn,
                                            "Track ID": det.get("track_id", det["face_id"] + 1),
                                            "Total Faces": data["total_faces"],
                                            "Emotion": det["emotion_label"],
                                            "Emotion Conf": f"{det['emotion_confidence']*100:.1f}%",
                                            "Face Conf": f"{det['face_confidence']*100:.1f}%",
                                            "Bounding Box": str(det["bbox"])
                                        })
                        except Exception as e:
                            st.error(f"Failed to process image '{fn}': {e}")

            # Save batch results to session state for persistent grid display
            st.session_state.batch_results = annotated_results
            st.session_state.batch_csv = csv_records

    # Display Batch Grid Outputs (4 Images Per Row)
    if getattr(st.session_state, "batch_results", None):
        annotated_results = st.session_state.batch_results
        csv_records = getattr(st.session_state, "batch_csv", [])

        st.markdown(f"#### 🖼️ Batch Annotated Outputs ({len(annotated_results)} Images, 4 Per Row)")
        
        # Zip download containing all annotated images
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in annotated_results:
                ann_b = render_predictions_bytes(item["image_bytes"], item["detections"], format="JPEG")
                safe_name = f"annotated_{os.path.basename(item['filename'])}"
                zout.writestr(safe_name, ann_b)
        zip_buf.seek(0)

        st.download_button(
            label="📦 Download All Annotated Images (.zip)",
            data=zip_buf,
            file_name="batch_annotated_images.zip",
            mime="application/zip"
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # 4 Images Per Row Grid Display
        for row_idx in range(0, len(annotated_results), 4):
            row_items = annotated_results[row_idx:row_idx+4]
            cols = st.columns(4)

            for col_idx, item in enumerate(row_items):
                with cols[col_idx]:
                    fn = os.path.basename(item["filename"])
                    img_b = item["image_bytes"]
                    dets = item["detections"]
                    num_faces = item["total_faces"]

                    ann_img = render_predictions(img_b, dets)
                    caption_str = f"📄 {fn} ({num_faces} faces)" if num_faces > 0 else f"📄 {fn} (No Face ≥ 65%)"
                    st.image(ann_img, caption=caption_str, use_container_width=True)
                    
                    # Individual Download Button for each image in grid
                    dl_bytes = render_predictions_bytes(img_b, dets, format="JPEG")
                    st.download_button(
                        label="📥 Download Image",
                        data=dl_bytes,
                        file_name=f"annotated_{fn}",
                        mime="image/jpeg",
                        key=f"dl_grid_{row_idx}_{col_idx}_{fn}",
                        use_container_width=True
                    )

        if csv_records:
            st.markdown("---")
            df_batch = pd.DataFrame(csv_records)
            st.markdown("#### 📋 Batch Summary Table")
            st.dataframe(df_batch, use_container_width=True)

            csv_bytes = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Batch Predictions CSV",
                data=csv_bytes,
                file_name="batch_emotion_predictions.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------------
# TAB 3: Video Analysis (1-in-3 Subsampling, Bounding Boxes, & Download)
# -------------------------------------------------------------------
with tab3:
    st.markdown("### 🎞️ Video Processing & Persistent Face Tracking")
    st.markdown("Subsamples **every 3rd frame** ($N=3$), renders **bounding boxes, track IDs, and emotion labels**, and provides an **annotated video download**.")

    uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"], key="video_upload")

    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🎬 Run Video Tracking & Emotion Pipeline", type="primary"):
            video_bytes = uploaded_video.getvalue()
            with st.spinner("Processing strided video frames with face tracking & rendering output video..."):
                try:
                    files = {"file": (uploaded_video.name, video_bytes, "video/mp4")}
                    params = {
                        "frame_stride": VIDEO_FRAME_STRIDE,
                        "conf_thresh": FACE_CONF_THRESH,
                        "emotion_conf_thresh": EMOTION_CONF_THRESH,
                        "margin": FACE_CROP_MARGIN
                    }
                    resp = requests.post(f"{BACKEND_URL}/predict-video", params=params, files=files, timeout=180)

                    if resp.status_code == 200:
                        v_data = resp.json()
                        st.success(f"Successfully processed {v_data['total_frames_processed']} strided frames in {v_data['execution_time_ms']} ms!")

                        # Render Annotated Output Video & Download Button
                        if v_data.get("annotated_video_b64"):
                            out_vid_bytes = base64.b64decode(v_data["annotated_video_b64"])
                            st.session_state.processed_video_bytes = out_vid_bytes
                            st.session_state.processed_video_name = uploaded_video.name
                        else:
                            st.session_state.processed_video_bytes = None

                        # Timeline Dataframe
                        rows = []
                        for item in v_data["frame_timeline"]:
                            f_idx = item["frame"]
                            t_sec = item["timestamp_sec"]
                            if item["total_faces"] == 0:
                                rows.append({"Frame": f_idx, "Time (s)": t_sec, "Track ID": "-", "Emotion": "NO_FACE_DETECTED", "Emotion Conf": "-", "Face Conf": "-"})
                            else:
                                for det in item["detections"]:
                                    rows.append({
                                        "Frame": f_idx,
                                        "Time (s)": t_sec,
                                        "Track ID": det.get("track_id", det["face_id"] + 1),
                                        "Emotion": det["emotion_label"],
                                        "Emotion Conf": f"{det['emotion_confidence']*100:.1f}%",
                                        "Face Conf": f"{det['face_confidence']*100:.1f}%"
                                    })

                        st.session_state.video_df_rows = rows
                    else:
                        st.error(f"Video API Error: {resp.text}")
                except Exception as e:
                    st.error(f"Video processing failed: {e}")

    # Display Processed Video & Download Option
    if getattr(st.session_state, "processed_video_bytes", None):
        st.markdown("---")
        st.markdown("#### 🎬 Annotated Output Video (Bounding Boxes, Track IDs, & Emotion Labels)")
        st.video(st.session_state.processed_video_bytes)

        st.download_button(
            label="📥 Download Annotated Video (.mp4)",
            data=st.session_state.processed_video_bytes,
            file_name=f"annotated_{st.session_state.processed_video_name}",
            mime="video/mp4",
            type="primary",
            use_container_width=True
        )

        if getattr(st.session_state, "video_df_rows", None):
            df_vid = pd.DataFrame(st.session_state.video_df_rows)
            st.markdown("#### ⏱️ Frame Timeline & Persistent Track History")
            st.dataframe(df_vid, use_container_width=True)

            csv_v = df_vid.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Video Timeline CSV", data=csv_v, file_name="video_tracked_timeline.csv", mime="text/csv")

# -------------------------------------------------------------------
# TAB 4: Live Camera Stream (Live Feed, Bbox Overlay, & Video Download)
# -------------------------------------------------------------------
with tab4:
    st.markdown("### 🎥 Live Continuous Camera Stream")
    st.markdown("Displays a **real-time live stream** from your webcam in HD. Live emotion events are logged below.")

    class EmotionVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_img_bgr = None
            self.log_queue = queue.Queue()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Subsample frames to maintain high framerate and avoid overloading backend
            if self.frame_count % VIDEO_FRAME_STRIDE == 0 or self.last_img_bgr is None:
                success, buffer = cv2.imencode('.jpg', img_bgr)
                if success:
                    try:
                        # Direct HTTP call to the backend
                        resp = call_predict_api(buffer.tobytes(), "stream.jpg", is_stream=True)
                        if resp.status_code == 200:
                            data = resp.json()
                            dets = data.get("detections", [])
                            
                            # Safely put detections into queue for the main thread to read
                            if len(dets) == 0:
                                self.log_queue.put({
                                    "Frame": self.frame_count,
                                    "Track ID": "-",
                                    "Emotion": "No Face Detected",
                                    "Emotion Conf": "-",
                                    "Face Conf": "-"
                                })
                            else:
                                for det in dets:
                                    self.log_queue.put({
                                        "Frame": self.frame_count,
                                        "Track ID": det.get("track_id", det.get("face_id", 0) + 1),
                                        "Emotion": det["emotion_label"],
                                        "Emotion Conf": f"{det['emotion_confidence']*100:.1f}%",
                                        "Face Conf": f"{det['face_confidence']*100:.1f}%"
                                    })

                            # Re-render the bounding boxes
                            ann_pil = render_predictions(buffer.tobytes(), dets)
                            img_rgb = np.array(ann_pil)
                            self.last_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        else:
                            self.last_img_bgr = img_bgr
                    except Exception:
                        self.last_img_bgr = img_bgr
            
            out_img = self.last_img_bgr if self.last_img_bgr is not None else img_bgr
            return av.VideoFrame.from_ndarray(out_img, format="bgr24")

    # The webrtc_streamer will expand to fill the container width and request 720p HD
    webrtc_ctx = webrtc_streamer(
        key="emotion-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}, "frameRate": {"ideal": 60}}, "audio": False},
        video_html_attrs={
            "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #3B82F6", "border-radius": "10px"},
            "autoPlay": True,
            "controls": False,
            "playsinline": True,
            "muted": True
        },
        async_processing=True
    )

    st.markdown("---")
    st.markdown("#### 📡 Live Event Logs")
    
    log_placeholder = st.empty()
    download_placeholder = st.empty()

    if "live_logs" not in st.session_state:
        st.session_state.live_logs = []

    if webrtc_ctx.state.playing:
        while True:
            if webrtc_ctx.video_processor:
                try:
                    # Pop logs from the background thread processing video
                    log_entry = webrtc_ctx.video_processor.log_queue.get(timeout=1.0)
                    st.session_state.live_logs.append(log_entry)
                    
                    # Update UI with the latest 10 logs so it doesn't freeze the browser
                    latest_logs = st.session_state.live_logs[-10:]
                    df_logs = pd.DataFrame(latest_logs)
                    log_placeholder.dataframe(df_logs, use_container_width=True)
                except queue.Empty:
                    pass
            else:
                break
    
    # Render Download Button if logs exist (even if stream is stopped)
    if not webrtc_ctx.state.playing and len(st.session_state.live_logs) > 0:
        df_all = pd.DataFrame(st.session_state.live_logs)
        log_placeholder.dataframe(df_all.tail(15), use_container_width=True)
        
        csv_v = df_all.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Complete Live Stream Logs (.csv)",
            data=csv_v,
            file_name="live_stream_logs.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
