import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime
from middleware.auth_middleware import token_required, JWT_SECRET
from config.db import detections_collection
from detect_audio import predict_audio

detect_bp = Blueprint("detect", __name__)

# Upload directory path
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

@detect_bp.route("/api/detect", methods=["POST"])
def detect():
    user_id = None
    # Check for authentication token to optionally save history
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        if token in ["null", "undefined"]:
            token = None
    else:
        token = request.cookies.get("token")
        
    if token:
        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            user_id = decoded.get("user_id")
        except Exception:
            pass
    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "No file uploaded"}), 400

    file = request.files["file"]
    # Sanitize filename and add unique identifier to prevent collisions and path traversal
    base_filename = secure_filename(file.filename) if file.filename else "upload.wav"
    unique_filename = f"{uuid.uuid4().hex}_{base_filename}"
    
    # Ensure uploads directory exists with safe permissions
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    file_path = os.path.join(UPLOADS_DIR, unique_filename)

    try:
        file.save(file_path)
    except Exception as e:
        current_app.logger.exception("Failed to save uploaded file")
        return jsonify({"error": "save_failed", "message": "Failed to save uploaded file"}), 500

    try:
        prediction = predict_audio(file_path)
        
        # Save history if user is authenticated
        if user_id:
            record = {
                "user_id": user_id,
                "filename": base_filename,
                "result": prediction["result"],
                "confidence": prediction["confidence"],
                "confidence_level": prediction.get("confidence_level", "Unknown"),
                "signals": prediction.get("signals", []),
                "timestamp": datetime.utcnow()
            }
            detections_collection.insert_one(record)
            
        return jsonify({
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "confidence_level": prediction.get("confidence_level", "Unknown"),
            "signals": prediction.get("signals", []),
        }), 200
    except Exception as e:
        current_app.logger.exception("Error during detection")
        return jsonify({"error": "detection_failed", "message": "An internal error occurred during detection"}), 500
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            current_app.logger.exception("Failed to remove uploaded file")

@detect_bp.route("/api/history", methods=["GET"])
@token_required
def get_history():
    try:
        user_id = request.user.get("user_id")
        # Fetch descending by timestamp
        cursor = detections_collection.find({"user_id": user_id}).sort("timestamp", -1)
        history = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "timestamp" in doc:
                doc["timestamp"] = doc["timestamp"].isoformat()
            history.append(doc)
            
        return jsonify({"history": history}), 200
    except Exception as e:
        current_app.logger.exception("Error fetching history")
        return jsonify({"error": "history_fetch_failed", "message": "Failed to fetch detection history"}), 500
