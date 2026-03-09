import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from middleware.auth_middleware import token_required
from detect_audio import predict_audio

detect_bp = Blueprint("detect", __name__)

@detect_bp.route("/api/detect", methods=["POST"])
@token_required
def detect():
    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "No file uploaded"}), 400

    file = request.files["file"]
    filename = secure_filename(file.filename) if file.filename else "upload.wav"
    file_path = os.path.join("uploads", filename)

    try:
        file.save(file_path)
    except Exception as e:
        current_app.logger.exception("Failed to save uploaded file")
        return jsonify({"error": "save_failed", "details": "An internal error occurred while saving the file"}), 500

    try:
        prediction = predict_audio(file_path)
        return jsonify({
            "result": prediction["result"],
            "confidence": prediction["confidence"]
        }), 200
    except Exception as e:
        current_app.logger.exception("Error during detection")
        return jsonify({"error": "detection_failed", "details": "An internal error occurred during detection"}), 500
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            current_app.logger.exception("Failed to remove uploaded file")
