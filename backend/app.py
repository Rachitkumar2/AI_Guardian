import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import NotFound
from dotenv import load_dotenv

# Import blueprints
from auth.auth_routes import auth_bp
from routes.detect_routes import detect_bp
from routes.profile_routes import profile_bp
from routes.security_routes import security_bp

# Load environment variables
load_dotenv()

# In production, serve React build from ../frontend/build
app = Flask(__name__, static_folder='../frontend/build', static_url_path='')

# CORS with credentials support
CORS(app, supports_credentials=True, origins=[
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5000",
    "http://192.168.29.107:3000",
    "http://192.168.29.107:5173",
    "https://ai-guardian-sigma.vercel.app"
])

# Ensure uploads directory exists
os.makedirs("uploads", exist_ok=True)

# Register Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(detect_bp)
app.register_blueprint(profile_bp)
app.register_blueprint(security_bp)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')

# Catch-all route to serve React app for client-side routing
@app.errorhandler(404)
def not_found(e):
    # Return JSON 404 for API and auth routes
    if request.path.startswith("/api/") or request.path.startswith("/auth/"):
        return jsonify({"error": "not_found", "message": "API endpoint not found"}), 404
    # Try to serve index.html for client-side routing, fallback to JSON 404
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except (NotFound, TypeError):
        return jsonify({"error": "not_found", "message": "Resource not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
