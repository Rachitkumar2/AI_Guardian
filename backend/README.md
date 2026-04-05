---
title: AI Guardian Backend
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Guardian Backend (Production)

This is the production backend for the **AI Guardian** platform, hosted on Hugging Face Spaces.

## 🚀 Features
- **Real-time Deepfake Audio Detection**: Uses a pretrained HuggingFace wav2vec2 model.
- **JWT Authentication**: Secure user management and session tracking.
- **MongoDB Atlas Integration**: Persistent data storage for detections and user profiles.
- **Flask Framework**: Scalable and lightweight API structure.

## ⚙️ Environment Variables
The following secrets must be configured in your Hugging Face Space Settings:
- `MONGO_URI`: Your MongoDB Atlas connection string.
- `JWT_SECRET`: A secure random string for signing tokens.
- `GOOGLE_CLIENT_ID`: Required for Google OAuth integrations.
- `ALLOWED_ORIGINS`: Comma-separated list of allowed frontend URLs (e.g., your Vercel URL).

## 🛠️ Deployment
This space is automatically built from the `backend/` directory using the provided `Dockerfile`. It uses the **16GB CPU** hardware tier for optimal performance during audio processing.
