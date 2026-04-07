import os
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import NotFound
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

# Import blueprints
from auth.auth_routes import auth_bp
from routes.detect_routes import detect_bp
from routes.profile_routes import profile_bp
from routes.security_routes import security_bp

# Load environment variables
load_dotenv()

# In production, we only provide API services (Frontend is on Vercel)
app = Flask(__name__)

# Hugging Face uses a proxy, so we must trust it to get the correct client IP.
# x_for=1 tells the app to trust the first proxy in the chain.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# CORS with credentials support
# Build origins list from env var + defaults for local dev
_allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5000",
    "http://192.168.29.107:3000",
    "http://192.168.29.107:5173",
    "https://ai-guardian-sigma.vercel.app",
]
# Add any extra origins from ALLOWED_ORIGINS env var (comma-separated)
_extra = os.environ.get("ALLOWED_ORIGINS", "")
if _extra:
    _allowed_origins.extend([o.strip() for o in _extra.split(",") if o.strip()])

CORS(app, supports_credentials=True, origins=_allowed_origins)

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
    """API Health Check"""
    return jsonify({
        "status": "online",
        "message": "AI Guardian API is running",
        "version": "1.0.1",
        "detected_ip": request.remote_addr,
        "endpoints": ["/api/detect", "/auth/login", "/auth/signup"]
    })

# Catch-all route for 404s
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not_found", "message": "API endpoint not found"}), 404


if __name__ == "__main__":
    # Hugging Face Spaces and many other platforms inject PORT environment variable.
    # We default to 7860 if not specified, which is the standard for HF Spaces.
    port = int(os.environ.get("PORT", 7860))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
