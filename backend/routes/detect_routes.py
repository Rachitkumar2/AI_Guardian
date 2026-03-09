import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from middleware.auth_middleware import token_required
from detect_audio import predict_audio

detect_bp = Blueprint("detect", __name__)

# Upload directory path
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

@detect_bp.route("/api/detect", methods=["POST"])
@token_required
def detect():
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
        return jsonify({
            "result": prediction["result"],
            "confidence": prediction["confidence"]
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
