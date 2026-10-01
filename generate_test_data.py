import os
import io
import zipfile
import numpy as np
import cv2
from PIL import Image

def generate_test_data():
    base_dir = os.path.join(os.path.dirname(__file__), "backend", "tests", "test_data")
    img_dir = os.path.join(base_dir, "images")
    zip_dir = os.path.join(base_dir, "zips")
    vid_dir = os.path.join(base_dir, "videos")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    # 1. Generate Sample Image
    img_path = os.path.join(img_dir, "sample_face.jpg")
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(img_path, format="JPEG")

    # 2. Generate Sample ZIP
    zip_path = os.path.join(zip_dir, "sample_batch.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(img_path, arcname="sample_face_1.jpg")
        z.write(img_path, arcname="nested/sample_face_2.jpg")

    # 3. Generate Sample Video (OpenCV)
    vid_path = os.path.join(vid_dir, "sample_stream.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vid_path, fourcc, 30.0, (640, 480))
    
    # Write 30 frames (1 second of video)
    for i in range(30):
        # Create a basic synthetic frame that changes slightly
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50 + i * 5, 200)
        out.write(frame)
    out.release()

    print(f"Test data generated successfully at: {base_dir}")

if __name__ == "__main__":
    generate_test_data()
