import os
import uuid
import requests
from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime

from middleware.auth_middleware import token_required, JWT_SECRET
from config.db import detections_collection
from models.usage_model import allowed_to_scan, increment_usage, FREE_GUEST_SCAN_LIMIT
from detect_audio import predict_audio

# Import new utilities
from utils.audio_utils import download_audio, convert_to_standard_wav

detect_bp = Blueprint("detect", __name__)

# Upload directory path
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

def _get_client_ip():
    # Since we added ProxyFix middleware in app.py, 
    # request.remote_addr will now correctly reflect the client's real IP 
    # through Hugging Face's proxy.
    return request.remote_addr or "unknown"

def _extract_guest_id(payload=None):
    if payload and payload.get("guest_id"):
        return payload.get("guest_id")

    return (
        request.headers.get("X-Guest-Id")
        or request.form.get("guest_id")
        or request.cookies.get("guest_id")
    )

@detect_bp.route("/api/detect", methods=["POST"])
def detect():
    user_id = None
    request_payload = request.get_json(silent=True) if request.is_json else None

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

    guest_id = _extract_guest_id(request_payload)
    guest_ip = _get_client_ip()
    guest_scans_used = 0

    if not user_id:
        # Keep a stable identifier for browser-side persistence.
        if not guest_id:
            guest_id = uuid.uuid4().hex

        usage_state = allowed_to_scan(guest_id=guest_id, guest_ip=guest_ip)
        guest_scans_used = usage_state["scans_used"]

        if not usage_state["allowed"]:
            return jsonify({
                "error": "limit_reached",
                "message": "Free limit reached. Please login to continue.",
                "requires_login": True,
                "free_limit": FREE_GUEST_SCAN_LIMIT,
                "scans_used": guest_scans_used,
                "scans_remaining": 0,
                "guest_id": guest_id,
            }), 403

    # Define cleanup registry
    temp_files = []
    file_path = None
    base_filename = "upload.wav"

    try:
        # CASE 1: JSON URL Input
        if request_payload and "url" in request_payload:
            url = request_payload["url"]
            if not url:
                return jsonify({"error": "invalid_url", "message": "URL cannot be empty"}), 400
            
            # Download the audio
            download_path = download_audio(url, UPLOADS_DIR)
            temp_files.append(download_path)
            
            # Convert to standard WAV (16kHz, mono)
            file_path = convert_to_standard_wav(download_path, UPLOADS_DIR)
            temp_files.append(file_path)
            
            # Track original filename for history
            base_filename = os.path.basename(url.split('?')[0])

        # CASE 2: Multipart File Upload
        elif "file" in request.files:
            file = request.files["file"]
            # Sanitize filename and add unique identifier to prevent collisions and path traversal
            base_filename = secure_filename(file.filename) if file.filename else "upload.wav"
            unique_filename = f"{uuid.uuid4().hex}_{base_filename}"
            
            # Ensure uploads directory exists with safe permissions
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            file_path = os.path.join(UPLOADS_DIR, unique_filename)
            file.save(file_path)
            temp_files.append(file_path)

        else:
            return jsonify({"error": "no_input", "message": "No file or URL provided"}), 400


        # Perform Detection using the existing AI pipeline
        prediction = predict_audio(file_path)

        # Save detection record
        record = {
            "user_id": user_id,
            "guest_id": guest_id if not user_id else None,
            "guest_ip": guest_ip if not user_id else None,
            "filename": base_filename,
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "confidence_level": prediction.get("confidence_level", "Unknown"),
            "signals": prediction.get("signals", []),
            "timestamp": datetime.utcnow()
        }
        detections_collection.insert_one(record)

        if not user_id:
            increment_usage(guest_id=guest_id, guest_ip=guest_ip)
            
        response_data = {
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "confidence_level": prediction.get("confidence_level", "Unknown"),
            "signals": prediction.get("signals", []),
            "free_limit": FREE_GUEST_SCAN_LIMIT if not user_id else None,
            "scans_used": (guest_scans_used + 1) if not user_id else None,
            "scans_remaining": max(0, FREE_GUEST_SCAN_LIMIT - (guest_scans_used + 1)) if not user_id else None,
        }

        if not user_id:
            response_data["guest_id"] = guest_id
        
        response = make_response(jsonify(response_data), 200)
        
        return response

    except ValueError as val_err:
        return jsonify({"error": "validation_failed", "message": str(val_err)}), 400
    except Exception as e:
        current_app.logger.exception("Error during detection")
        return jsonify({"error": "detection_failed", "message": str(e)}), 500
    finally:
        # Cleanup all registered temp files
        for temp_path in temp_files:
            try:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                current_app.logger.exception(f"Failed to remove temp file: {temp_path}")


@detect_bp.route("/api/free-scan-status", methods=["GET"])
def free_scan_status():
    user_id = None
    guest_id = None

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
            user_id = None

    if user_id:
        return jsonify({
            "requires_login": False,
            "is_authenticated": True,
            "free_limit": None,
            "scans_used": None,
            "scans_remaining": None,
            "is_locked": False,
            "guest_id": None,
        }), 200

    guest_id = _extract_guest_id()
    if not guest_id:
        guest_id = uuid.uuid4().hex

    guest_ip = _get_client_ip()
    usage_state = allowed_to_scan(guest_id=guest_id, guest_ip=guest_ip)
    scans_used = usage_state["scans_used"]
    scans_remaining = usage_state["scans_remaining"]

    return jsonify({
        "requires_login": not usage_state["allowed"],
        "is_authenticated": False,
        "free_limit": FREE_GUEST_SCAN_LIMIT,
        "scans_used": scans_used,
        "scans_remaining": scans_remaining,
        "is_locked": not usage_state["allowed"],
        "guest_id": guest_id,
    }), 200

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
                doc["timestamp"] = doc["timestamp"].isoformat() + "Z"
            history.append(doc)
            
        return jsonify({"history": history}), 200
    except Exception as e:
        current_app.logger.exception("Error fetching history")
        return jsonify({"error": "history_fetch_failed", "message": "Failed to fetch detection history"}), 500
