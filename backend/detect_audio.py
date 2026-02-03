import os
import logging
import torch
import torch.nn as nn
import librosa
import numpy as np
import requests
import pickle

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.getcwd(), "audio_model.pth"))
SCALER_PATH = os.environ.get("SCALER_PATH", os.path.join(os.getcwd(), "scaler.pkl"))
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

# Enhanced Feature Extraction (must match training)
def extract_features(file_path, sr=16000, n_mfcc=40):
    """Extract comprehensive audio features for deepfake detection."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=5)
        
        # Pad or truncate audio to consistent length
        target_length = sr * 5
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # 1. MFCCs (40 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
        
        # 2. Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)
        
        # 3. Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # 4. RMS Energy
        rms = np.mean(librosa.feature.rms(y=audio))
        
        # 5. Chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
        
        # 6. Mel spectrogram statistics
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        mel_mean = np.mean(mel_spec, axis=1)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,           # 40 features
            mfcc_std,            # 40 features
            mfcc_delta,          # 40 features
            [spectral_centroid], # 1 feature
            [spectral_bandwidth],# 1 feature
            [spectral_rolloff],  # 1 feature
            spectral_contrast,   # 7 features
            [zcr],               # 1 feature
            [rms],               # 1 feature
            chroma,              # 12 features
            mel_mean             # 40 features
        ])
        
        return features
        
    except Exception as e:
        logger.exception(f"Error extracting features from {file_path}")
        raise RuntimeError(f"Feature extraction failed: {e}")

# Enhanced model architecture (must match training)
class AudioClassifier(nn.Module):
    def __init__(self, input_size=184):
        super(AudioClassifier, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

# load model and scaler once at import time
model = None
scaler = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = AudioClassifier()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Model loaded from {MODEL_PATH}")
    
    # Load scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {SCALER_PATH}")
    else:
        logger.warning(f"Scaler not found at {SCALER_PATH}. Predictions may be inaccurate.")
        
except FileNotFoundError as e:
    model = None
    logger.exception(e)
except Exception:
    model = None
    logger.exception("Failed to load model (unexpected).")

def predict_audio(file_path):
    """
    Returns dict with "result" ("Real"/"Fake") and "confidence" (0-100%).
    Raises exceptions with descriptive messages on failure.
    """
    if model is None:
        raise RuntimeError("Model is not loaded on server. Set MODEL_PATH or MODEL_URL.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Extract features
    features = extract_features(file_path)
    
    # Normalize features if scaler is available
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    tensor = torch.tensor(features, dtype=torch.float32)

    try:
        with torch.no_grad():
            out = model(tensor)
            probabilities = torch.softmax(out, dim=1)
            pred = int(torch.argmax(out, dim=1).item())
            confidence = float(probabilities[0][pred].item()) * 100
        
        result = "Fake" if pred == 1 else "Real"
        return {
            "result": result,
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        logger.exception("Inference failed")
        raise RuntimeError(f"Inference failed: {e}")
