import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import blueprints
from auth.auth_routes import auth_bp
from routes.detect_routes import detect_bp

# Load environment variables
load_dotenv()

# In production, serve React build from ../frontend/build
app = Flask(__name__, static_folder='../frontend/build', static_url_path='')

# CORS with credentials support for HTTP-only cookies
CORS(app, supports_credentials=True, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5000"
])

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(detect_bp)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

# Catch-all route to serve React app for client-side routing
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "not_found", "message": "API endpoint not found"}), 404
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
