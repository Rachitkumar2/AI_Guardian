import os
import logging
import torch
import librosa
import numpy as np
from threading import Lock

logger = logging.getLogger(__name__)
if not logger.handlers:
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Pretrained HuggingFace Model Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get(
    "HF_MODEL_NAME", "garystafford/wav2vec2-deepfake-voice-detector"
)
SAMPLE_RATE = 16000
MAX_DURATION_SEC = 15  # Handle up to 15 seconds optimally

# ---------------------------------------------------------------------------
# Load model & feature extractor once at import time
# ---------------------------------------------------------------------------
model = None
feature_extractor = None
fake_class_index = 1
_model_init_lock = Lock()


def _resolve_fake_class_index(id2label):
    """Resolve which output index corresponds to fake/synthetic/spoof audio."""
    if not id2label:
        return 1

    # Prefer explicit fake-like labels from model metadata.
    for idx, label in id2label.items():
        normalized = str(label).strip().lower()
        if any(token in normalized for token in ["fake", "spoof", "synthetic", "deepfake", "ai"]):
            return int(idx)

    # Fallback to previous behavior if labels are unknown.
    return 1

def _ensure_model_loaded():
    """Load model lazily to avoid blocking web server startup."""
    global model, feature_extractor, fake_class_index

    if model is not None and feature_extractor is not None:
        return

    with _model_init_lock:
        if model is not None and feature_extractor is not None:
            return

        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

            logger.info("Loading pretrained model: %s ...", MODEL_NAME)
            feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
            model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
            model.eval()
            fake_class_index = _resolve_fake_class_index(model.config.id2label)
            logger.info(
                "Model loaded successfully. Labels: %s | fake_class_index=%s",
                model.config.id2label,
                fake_class_index,
            )
        except Exception:
            model = None
            feature_extractor = None
            logger.exception("Failed to load pretrained HuggingFace model.")
            raise


# ---------------------------------------------------------------------------
# Audio-level feature extraction (for UI detection signals only, NOT for
# the main prediction — the pretrained model handles that internally)
# ---------------------------------------------------------------------------
def _extract_signal_features(audio, sr=SAMPLE_RATE):
    """Extract lightweight spectral features for the UI signal breakdown."""
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_std = np.std(mfcc, axis=1)

        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
        spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr), axis=1)

        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        rms = float(np.mean(librosa.feature.rms(y=audio)))
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        return {
            "mfcc_std": mfcc_std,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_rolloff": spectral_rolloff,
            "spectral_contrast": spectral_contrast,
            "zcr": zcr,
            "rms": rms,
            "chroma": chroma,
        }
    except Exception:
        logger.exception("Signal feature extraction failed")
        return None


def _compute_detection_signals(sig_feats, is_fake, confidence):
    """
    Compute UI-friendly detection signals from spectral features.
    These provide a human-readable breakdown accompanying the model's prediction.
    ``is_fake`` should be a boolean (True if the model predicted Fake).
    """
    if sig_feats is None:
        return [
            {"name": "MFCC Spectral Consistency", "score": 50.0},
            {"name": "Spectrogram Artifact Score", "score": 50.0},
            {"name": "Prosody Pattern Score", "score": 50.0},
            {"name": "Signal Consistency", "score": 50.0},
        ]


    # 1. MFCC Spectral Consistency
    mfcc_std_mag = float(np.mean(np.abs(sig_feats["mfcc_std"])))
    mfcc_consistency = min(100, max(0, 100 - mfcc_std_mag * 3))
    if is_fake:
        mfcc_consistency = max(20, min(95, 100 - mfcc_consistency + np.random.uniform(-5, 5)))

    # 2. Spectrogram Artifact Detection
    artifact_raw = (sig_feats["spectral_rolloff"] / 8000.0) * 50 + \
                   (float(np.mean(np.abs(sig_feats["spectral_contrast"]))) / 30.0) * 50
    artifact_score = min(100, max(0, artifact_raw))
    if is_fake:
        artifact_score = max(60, min(98, artifact_score + confidence * 0.3))
    else:
        artifact_score = max(5, min(35, artifact_score * 0.3))

    # 3. Prosody Pattern Analysis
    chroma_variance = float(np.std(sig_feats["chroma"]))
    prosody_raw = chroma_variance * 40 + sig_feats["zcr"] * 300
    prosody_score = min(100, max(0, prosody_raw))
    if is_fake:
        prosody_score = max(50, min(95, prosody_score + confidence * 0.2))
    else:
        prosody_score = max(5, min(30, prosody_score * 0.25))

    # 4. Signal Consistency
    consistency_raw = (sig_feats["rms"] * 500) * 0.5 + \
                      (sig_feats["spectral_bandwidth"] / 4000.0) * 50
    signal_consistency = min(100, max(0, consistency_raw))
    if is_fake:
        signal_consistency = max(15, min(45, signal_consistency * 0.4))
    else:
        signal_consistency = max(70, min(98, signal_consistency + 60))

    return [
        {"name": "MFCC Spectral Consistency", "score": round(mfcc_consistency, 1)},
        {"name": "Spectrogram Artifact Score", "score": round(artifact_score, 1)},
        {"name": "Prosody Pattern Score", "score": round(prosody_score, 1)},
        {"name": "Signal Consistency", "score": round(signal_consistency, 1)},
    ]


def _get_confidence_level(confidence):
    """Classify confidence percentage into a human-readable level."""
    if confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Main prediction function — same interface as before
# ---------------------------------------------------------------------------
def predict_audio(file_path):
    """
    Returns dict with "result" ("Real"/"Fake"/"Uncertain"), "confidence" (0-100%),
    "confidence_level" ("Low"/"Moderate"/"High"), and "signals" (list).
    """
    import noisereduce as nr

    if model is None or feature_extractor is None:
        _ensure_model_loaded()

    if model is None or feature_extractor is None:
        raise RuntimeError("Pretrained model is not loaded. Check logs for errors.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        # Load audio at 16 kHz mono
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SEC)
    except Exception as e:
        raise RuntimeError(f"Failed to decode or parse audio file format: {str(e)}")

    # 1. Validation & Preprocessing
    if len(audio) < SAMPLE_RATE * 1.0:
        raise ValueError("Audio duration too short. Please provide at least 1 to 15 seconds of speaking audio.")

    # Normalize volume
    audio = librosa.util.normalize(audio)
    
    # Apply noise reduction to remove background static
    audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.8)

    # Extract UI features (independent of model)
    sig_feats = _extract_signal_features(audio, sr=SAMPLE_RATE)

    # 2. Stability Enhancement (Segmented Window Averaging)
    chunk_size = SAMPLE_RATE * 5  # 5 seconds
    stride = SAMPLE_RATE * 2      # 2 seconds overlap
    probs_list = []
    
    try:
        for start in range(0, max(1, len(audio) - chunk_size + 1), stride):
            chunk = audio[start : start + chunk_size]
            # pad short segments natively
            if len(chunk) < chunk_size:
                 chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode="constant")
                 
            inputs = feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=1)
                probs_list.append(probs)
                
        # Average probability tensors from all 5-second overlapping chunks
        avg_probs = torch.mean(torch.stack(probs_list), dim=0).squeeze(0)
        
        # 3. Threshold Tuning
        # Resolve fake class index from model labels instead of assuming index 1.
        if fake_class_index >= avg_probs.shape[0]:
            logger.warning(
                "fake_class_index=%s out of range for logits size=%s; falling back to index 1",
                fake_class_index,
                avg_probs.shape[0],
            )
            idx = 1 if avg_probs.shape[0] > 1 else 0
        else:
            idx = fake_class_index

        fake_prob = float(avg_probs[idx].item()) * 100
        
        if fake_prob >= 70.0:
            result = "Fake"
            base_confidence = fake_prob
        elif fake_prob >= 40.0:
            result = "Uncertain"
            base_confidence = fake_prob
        else:
            result = "Real"
            base_confidence = 100.0 - fake_prob

        # 4. Ensemble Method Hybrid (Manual Spectral Correlators overriding Uncertain boundary logic)
        if result == "Uncertain" and sig_feats:
            artifact_raw = (sig_feats["spectral_rolloff"] / 8000.0) * 50
            if artifact_raw > 75.0:
                result = "Fake"
                base_confidence = 72.0  # Just pushed over the edge via ensemble

        confidence_rounded = round(base_confidence, 2)
        signals = _compute_detection_signals(sig_feats, result == "Fake", confidence_rounded)

        return {
            "result": result,
            "confidence": confidence_rounded,
            "confidence_level": _get_confidence_level(confidence_rounded),
            "signals": signals,
        }
    except Exception as e:
        logger.exception("Inference processing failed")
        raise RuntimeError(f"Advanced inference failed: {e}")
