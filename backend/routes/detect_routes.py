import os
import uuid
import hashlib
import requests
from flask import Blueprint, request, jsonify, current_app, make_response
from werkzeug.utils import secure_filename
import jwt
from datetime import datetime

from middleware.auth_middleware import token_required, JWT_SECRET
from config.db import detections_collection
from detect_audio import predict_audio

# Import new utilities
from utils.audio_utils import download_audio, convert_to_standard_wav

detect_bp = Blueprint("detect", __name__)

# Upload directory path
UPLOADS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

# Maximum free scans for guest users
MAX_FREE_SCANS = 2


def _get_client_ip():
    trust_proxy_headers = os.environ.get("TRUST_PROXY_HEADERS", "0") == "1"
    forwarded_for = request.headers.get("X-Forwarded-For")
    if trust_proxy_headers and forwarded_for:
        # Only trust proxy headers when explicitly enabled by environment.
        return forwarded_for.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _get_guest_fingerprint():
    ip = _get_client_ip()
    user_agent = request.headers.get("User-Agent", "unknown")
    raw = f"{ip}|{user_agent}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

    # --- Guest scan limit (cookie-based, works in all browsers) ---
    guest_id = None
    guest_fingerprint = None
    guest_ip = None
    scans_used = 0
    scans_remaining = MAX_FREE_SCANS

    if not user_id:
        guest_id = request.cookies.get("guest_id")
        guest_ip = _get_client_ip()
        guest_fingerprint = _get_guest_fingerprint()
        
        if not guest_id:
            # New guest — generate an ID now (will be set as cookie in response)
            guest_id = uuid.uuid4().hex

        # Count by persistent guest cookie OR server-side fingerprint fallback.
        # This makes guest limits harder to bypass by clearing browser storage.
        scans_used = detections_collection.count_documents({
            "user_id": None,
            "$or": [
                {"guest_ip": guest_ip},
                {"guest_id": guest_id},
                {"guest_fingerprint": guest_fingerprint}
            ]
        })
        scans_remaining = max(0, MAX_FREE_SCANS - scans_used)

        if scans_used >= MAX_FREE_SCANS:
            return jsonify({
                "error": "limit_reached",
                "message": "You have reached your limit of 2 free scans. Please log in to continue.",
                "requires_login": True,
                "free_limit": MAX_FREE_SCANS,
                "scans_used": scans_used,
                "scans_remaining": 0
            }), 403

    # Define cleanup registry
    temp_files = []
    file_path = None
    base_filename = "upload.wav"

    try:
        # CASE 1: JSON URL Input
        if request.is_json and "url" in request.json:
            url = request.json["url"]
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
            "guest_id": guest_id,  # None for authenticated users
            "guest_ip": guest_ip,
            "guest_fingerprint": guest_fingerprint,
            "filename": base_filename,
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "confidence_level": prediction.get("confidence_level", "Unknown"),
            "signals": prediction.get("signals", []),
            "timestamp": datetime.utcnow()
        }
        detections_collection.insert_one(record)
            
        response_data = {
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "confidence_level": prediction.get("confidence_level", "Unknown"),
            "signals": prediction.get("signals", []),
            "free_limit": MAX_FREE_SCANS if not user_id else None,
            "scans_used": (scans_used + 1) if not user_id else None,
            "scans_remaining": max(0, scans_remaining - 1) if not user_id else None,
        }
        
        response = make_response(jsonify(response_data), 200)

        # Set guest_id cookie for anonymous users so we can track their scans
        if not user_id and guest_id:
            response.set_cookie(
                "guest_id",
                guest_id,
                max_age=60 * 60 * 24 * 365,  # 1 year
                httponly=True,
                samesite="Lax",
                path="/"
            )
        
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
            "is_locked": False
        }), 200

    guest_id = request.cookies.get("guest_id")
    guest_ip = _get_client_ip()
    guest_fingerprint = _get_guest_fingerprint()

    scans_used = detections_collection.count_documents({
        "user_id": None,
        "$or": [
            {"guest_ip": guest_ip},
            {"guest_id": guest_id},
            {"guest_fingerprint": guest_fingerprint}
        ]
    })
    scans_remaining = max(0, MAX_FREE_SCANS - scans_used)

    return jsonify({
        "requires_login": scans_used >= MAX_FREE_SCANS,
        "is_authenticated": False,
        "free_limit": MAX_FREE_SCANS,
        "scans_used": scans_used,
        "scans_remaining": scans_remaining,
        "is_locked": scans_used >= MAX_FREE_SCANS
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
