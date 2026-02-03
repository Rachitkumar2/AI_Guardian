from flask import Flask, request, jsonify, send_from_directory, current_app
from flask_cors import CORS
import os
from detect_audio import predict_audio

# In production, serve React build from ../frontend/build
app = Flask(__name__, static_folder='../frontend/build', static_url_path='')
CORS(app)  # Enable CORS for React development

# ensure uploads exists (gunicorn won't run __main__)
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def home():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/api/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "no_file", "message": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename or "upload.wav"
    file_path = os.path.join("uploads", filename)

    try:
        file.save(file_path)
    except Exception as e:
        current_app.logger.exception("Failed to save uploaded file")
        return jsonify({"error": "save_failed", "details": str(e)}), 500

    try:
        prediction = predict_audio(file_path)
        return jsonify({
            "result": prediction["result"],
            "confidence": prediction["confidence"]
        }), 200
    except Exception as e:
        current_app.logger.exception("Error during detection")
        return jsonify({"error": "detection_failed", "details": str(e)}), 500
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            current_app.logger.exception("Failed to remove uploaded file")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

# Catch-all route to serve React app for client-side routing
@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, 'index.html')
