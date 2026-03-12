# 🛡️ AI Guardian – Deepfake Voice Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/React-18.2-61DAFB?style=for-the-badge&logo=react" alt="React">
  <img src="https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

**AI Guardian** is a web-based tool designed to detect whether an audio file contains a **real human voice** or an **AI-generated deepfake**. Built with a PyTorch-based neural network and powered by a Flask API, it allows users to upload audio files and instantly get a prediction — *Real* or *Fake* with confidence scores.

---

## 🚀 Features

- 🎤 Upload `.wav`, `.mp3`, `.flac` audio via drag & drop or file picker
- 🔍 Detects AI-generated (deepfake) vs. Real human voice
- 🧠 Deep Neural Network trained on **184 audio features**
- 📊 Confidence score with each prediction
- 📦 Flask backend API with PyTorch inference
- ⚛️ Modern React frontend with Tailwind CSS
- 🎨 Smooth animations with Framer Motion
- 🚀 Production-ready with Vite build system

---

## 🏗️ Project Structure

```
AI_Guardian/
├── backend/
│   ├── app.py                 # Flask API server
│   ├── detect_audio.py        # Audio detection & inference
│   ├── train_model.py         # Model training pipeline
│   ├── preprocess_audio.py    # Audio preprocessing utilities
│   ├── audio_model.pth        # Trained PyTorch model (download separately)
│   ├── scaler.pkl             # Feature scaler (download separately)
│   ├── Requirements.txt       # Python dependencies
│   ├── procfile               # Deployment config
│   └── data/
│       ├── real/              # Real voice samples for training
│       └── fake/              # AI-generated samples for training
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component
│   │   ├── main.jsx           # React entry point
│   │   ├── index.css          # Global styles
│   │   └── components/
│   │       ├── Header.jsx     # App header
│   │       ├── UploadZone.jsx # File upload component
│   │       ├── ResultDisplay.jsx  # Detection results
│   │       └── FeatureCards.jsx   # Feature highlights
│   ├── package.json           # Node dependencies
│   └── vite.config.js         # Vite configuration
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## 💡 How It Works

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│   Flask API     │────▶│  PyTorch Model  │
│   (Frontend)    │     │   /api/detect   │     │  (184 features) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         │                      ▼                       │
         │              ┌─────────────────┐            │
         │              │ Feature Extract │            │
         │              │   (librosa)     │            │
         │              └─────────────────┘            │
         │                      │                       │
         ▼                      ▼                       ▼
    Upload Audio ──▶ Extract 184 Features ──▶ Predict Real/Fake
```

### Feature Extraction (184 total features)

| Feature Type | Count | Description |
|--------------|-------|-------------|
| MFCC (mean, std, delta) | 120 | Spectral envelope characteristics |
| Spectral Features | 10 | Centroid, bandwidth, rolloff, contrast |
| Zero Crossing Rate | 1 | Voice naturalness indicator |
| RMS Energy | 1 | Amplitude patterns |
| Chroma Features | 12 | Harmonic content |
| Mel Spectrogram | 40 | Frequency distribution |

### Neural Network Architecture

```
Input (184) → Dense(256) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(128) → BatchNorm → ReLU → Dropout(0.3)
           → Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
           → Dense(32)  → BatchNorm → ReLU
           → Dense(2)   → Softmax → Output (Real/Fake)
```

---

## 🧪 Local Setup

### Prerequisites

- Python 3.8+
- Node.js 18+
- pip & npm

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Rachitkumar2/AI_Guardian
cd AI_Guardian
```

### 2️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r Requirements.txt
```

### 3️⃣ Download Model Files

> ⚠️ Model files are not included in the repo (too large). Download from:

| File | Download Link | Place in |
|------|---------------|----------|
| `audio_model.pth` | [Google Drive](#) | `backend/` |
| `scaler.pkl` | [Google Drive](#) | `backend/` |


### 4️⃣ Frontend Setup

```bash
cd frontend
npm install
```

### 5️⃣ Run the Application

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Open http://localhost:5173 in your browser.

---

## 🎓 Training Your Own Model

### Step 1: Get Training Data

**Real Human Voice:**
- [LibriSpeech](http://www.openslr.org/12/) - Download `dev-clean.tar.gz` (337MB)
- [Common Voice](https://commonvoice.mozilla.org/en/datasets)

**AI-Generated/Fake Voice:**
- [ASVspoof 2019](https://www.asvspoof.org/) - Best for deepfake detection
- [WaveFake](https://zenodo.org/record/5642694) - Direct download

### Step 2: Organize Data

```
backend/data/
├── real/
│   ├── sample1.wav
│   ├── sample2.flac
│   └── ...
└── fake/
    ├── fake1.wav
    ├── fake2.mp3
    └── ...
```

Recommended: **50-100+ samples per class** for good accuracy.

### Step 3: Train

```bash
cd backend
python train_model.py
```

This will generate:
- `audio_model.pth` - Trained model weights
- `scaler.pkl` - Feature normalization scaler

---

## 📡 API Reference

### POST `/api/detect`

Analyze an audio file for deepfake detection.

**Request:**
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/api/detect
```

**Response:**
```json
{
  "result": "Real",
  "confidence": 94.56
}
```

| Field | Type | Description |
|-------|------|-------------|
| `result` | string | `"Real"` or `"Fake"` |
| `confidence` | float | Confidence percentage (0-100) |

**Error Response:**
```json
{
  "error": "no_file",
  "message": "No file uploaded"
}
```

---

## 🗓️ Development Roadmap (3 Months)

### Month 1: Foundation & Data (Mar 2026)
- [x] Core backend API
- [x] React frontend with upload
- [x] PyTorch model architecture
- [ ] Host model on cloud storage
- [ ] Add model download script
- [ ] Unit tests & CI/CD pipeline
- [ ] Docker support

### Month 2: Features & UI (Apr 2026)
- [ ] Audio waveform visualization
- [ ] Confidence score bar
- [ ] Analysis history (local storage)
- [ ] Batch file upload
- [ ] Dark mode toggle
- [ ] PWA support

### Month 3: Polish & Deploy (May 2026)
- [ ] API documentation (OpenAPI)
- [ ] Model explainability
- [ ] Rate limiting
- [ ] Production deployment (Render/Heroku)
- [ ] Demo GIF & documentation
- [ ] v1.0.0 Release

---




## 🙏 Acknowledgments

- [LibriSpeech](http://www.openslr.org/12/) for real voice datasets
- [ASVspoof](https://www.asvspoof.org/) for deepfake audio research
- [librosa](https://librosa.org/) for audio feature extraction
- [PyTorch](https://pytorch.org/) for deep learning framework

---


