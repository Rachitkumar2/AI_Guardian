import os
import logging
import torch
import librosa
import numpy as np
import requests

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "audio_model.pth"))
MODEL_URL  = os.environ.get("MODEL_URL", None)  # optional: set on Render to download model

def download_model(url, dest):
    logger.info(f"Downloading model from {url} -> {dest}")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logger.info("Model downloaded.")

# attempt download if missing and URL provided
if not os.path.exists(MODEL_PATH) and MODEL_URL:
    try:
        download_model(MODEL_URL, MODEL_PATH)
    except Exception:
        logger.exception("Model download failed at startup.")

# model architecture (must match how it was trained)
class AudioClassifier(torch.nn.Module):
    def __init__(self, input_size=13):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# load model once at import time (map to CPU)
model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = AudioClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")
except FileNotFoundError as e:
    model = None
    logger.exception(e)
except Exception:
    model = None
    logger.exception("Failed to load model (unexpected).")

def predict_audio(file_path):
    """
    Returns "Real" or "Fake". Raises exceptions with descriptive messages on failure.
    """
    if model is None:
        raise RuntimeError("Model is not loaded on server. Set MODEL_PATH or MODEL_URL.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        logger.exception("librosa failed to load file")
        raise RuntimeError(f"Failed to load audio: {e}")

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        feat = np.mean(mfcc, axis=1)  # shape (13,)
        tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
    except Exception as e:
        logger.exception("Feature extraction failed")
        raise RuntimeError(f"Feature extraction failed: {e}")

    try:
        with torch.no_grad():
            out = model(tensor)
            pred = int(torch.argmax(out, dim=1).item())
        return "Fake" if pred == 1 else "Real"
    except Exception as e:
        logger.exception("Inference failed")
        raise RuntimeError(f"Inference failed: {e}")
