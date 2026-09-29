import os
import cv2
import numpy as np
import torch
from typing import Dict, Any
from .config import EMOTION_MODEL_PATH, EMOTIONS

class EmotionClassifier:
    def __init__(self, model_path: str = EMOTION_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.emotions = EMOTIONS
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                if hasattr(self.model, "names") and self.model.names:
                    if isinstance(self.model.names, dict):
                        self.emotions = list(self.model.names.values())
                    elif isinstance(self.model.names, list):
                        self.emotions = self.model.names
                self.is_loaded = True
            except Exception as e:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.model = torch.load(self.model_path, map_location=device)
                    if hasattr(self.model, "eval"):
                        self.model.eval()
                    if hasattr(self.model, "to"):
                        self.model.to(device)
                    self.device = device
                    self.is_loaded = True
                except Exception as ex:
                    print(f"Warning: Failed to load PyTorch emotion model from {self.model_path}: {ex}")
                    self.is_loaded = False
        else:
            print(f"Notice: Emotion model file not found at {self.model_path}. Classifier running in mock mode.")

    def classify(self, face_crop_bgr: np.ndarray) -> Dict[str, Any]:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return {
                "label": "Neutral",
                "cls_conf": 0.0,
                "all_scores": {emo: 0.0 for emo in self.emotions}
            }

        if self.is_loaded and self.model is not None:
            # Check if YOLO classification model
            if hasattr(self.model, "predict"):
                results = self.model.predict(face_crop_bgr, verbose=False)
                probs = results[0].probs
                top1 = int(probs.top1)
                label = self.emotions[top1] if top1 < len(self.emotions) else "Neutral"
                conf = float(probs.top1conf)
                scores = {self.emotions[i]: float(p) for i, p in enumerate(probs.data.tolist()) if i < len(self.emotions)}
                return {"label": label, "cls_conf": conf, "all_scores": scores}
            else:
                # Custom PyTorch model inference
                import torchvision.transforms as transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((48, 48)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                img_pil = Image.fromarray(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB))
                tensor = transform(img_pil).unsqueeze(0)
                if hasattr(self, "device"):
                    tensor = tensor.to(self.device)
                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    conf, predicted = torch.max(probs, 0)
                    top_idx = int(predicted.item())
                    label = self.emotions[top_idx] if top_idx < len(self.emotions) else "Neutral"
                    scores = {self.emotions[i]: float(probs[i]) for i in range(min(len(self.emotions), len(probs)))}
                    return {"label": label, "cls_conf": float(conf), "all_scores": scores}
        else:
            # Mock classifier for testing when model file is absent
            # Returns 'Happy' as dummy top emotion with 0.88 confidence
            scores = {emo: (0.88 if emo == "Happy" else 0.02) for emo in self.emotions}
            return {
                "label": "Happy",
                "cls_conf": 0.88,
                "all_scores": scores
            }
