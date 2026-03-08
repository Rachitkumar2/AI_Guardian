from flask import Flask, request, jsonify, send_from_directory, current_app, make_response
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import bcrypt
import jwt
from datetime import datetime, timedelta
from functools import wraps
from dotenv import load_dotenv
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from detect_audio import predict_audio

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

# ── MongoDB Connection ──────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/ai_guardian")
JWT_SECRET = os.environ.get("JWT_SECRET", "change-this-secret-in-production")

client = MongoClient(MONGO_URI)
db = client["ai_guardian"]
users_collection = db["users"]

# Create unique index on email
users_collection.create_index("email", unique=True)


# ── JWT Middleware ──────────────────────────────────────────────────
def token_required(f):
    """Decorator that reads JWT from HTTP-only cookie and verifies it."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get("token")

        if not token:
            return jsonify({
                "error": "unauthorized",
                "message": "Authentication required"
            }), 401

        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user = decoded
        except jwt.ExpiredSignatureError:
            return jsonify({
                "error": "token_expired",
                "message": "Token has expired. Please login again."
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                "error": "invalid_token",
                "message": "Invalid token"
            }), 401

        return f(*args, **kwargs)
    return decorated


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve React frontend"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({
            "error": "missing_fields",
            "message": "All fields are required"
        }), 400

    try:
        # Hash password with bcrypt
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Insert user into MongoDB
        user_doc = {
            "name": name,
            "email": email,
            "password": hashed,
            "created_at": datetime.utcnow()
        }
        result = users_collection.insert_one(user_doc)

        # Generate JWT token
        payload = {
            "user_id": str(result.inserted_id),
            "email": email,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

        # Set JWT in HTTP-only cookie and return user info
        response = make_response(jsonify({
            "message": "User created successfully",
            "user": {
                "id": str(result.inserted_id),
                "name": name,
                "email": email
            }
        }), 201)

        response.set_cookie(
            "token",
            token,
            httponly=True,
            samesite="Lax",
            max_age=86400
        )
        return response

    except Exception as e:
        if "duplicate key" in str(e).lower() or "E11000" in str(e):
            return jsonify({
                "error": "email_exists",
                "message": "Email already registered"
            }), 409
        current_app.logger.exception("Error during signup")
        return jsonify({
            "error": "server_error",
            "message": str(e)
        }), 500


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({
            "error": "missing_fields",
            "message": "Email and password are required"
        }), 400

    try:
        user = users_collection.find_one({"email": email})

        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            # Generate JWT token
            payload = {
                "user_id": str(user["_id"]),
                "email": user["email"],
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=24)
            }
            token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

            response = make_response(jsonify({
                "message": "Login successful",
                "user": {
                    "id": str(user["_id"]),
                    "name": user.get("name", ""),
                    "email": user["email"]
                }
            }), 200)

            response.set_cookie(
                "token",
                token,
                httponly=True,
                samesite="Lax",
                max_age=86400
            )
            return response
        else:
            return jsonify({
                "error": "invalid_credentials",
                "message": "Invalid email or password"
            }), 401

    except Exception as e:
        current_app.logger.exception("Error during login")
        return jsonify({
            "error": "server_error",
            "message": str(e)
        }), 500


@app.route("/api/google-login", methods=["POST"])
def google_login():
    data = request.json
    token = data.get("credential")
    access_token = data.get("access_token")
    
    if not token and not access_token:
        return jsonify({"error": "missing_token", "message": "Google token is required"}), 400

    try:
        email = None
        name = None
        
        if access_token:
            import requests
            resp = requests.get(f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={access_token}")
            if resp.status_code != 200:
                return jsonify({"error": "invalid_token", "message": "Invalid Google access token"}), 400
            user_info = resp.json()
            email = user_info.get("email")
            name = user_info.get("name")
        elif token:
            # Verify the Google token
            client_id = os.environ.get("GOOGLE_CLIENT_ID")
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
            email = idinfo.get("email")
            name = idinfo.get("name")
        
        if not email:
            return jsonify({"error": "invalid_token", "message": "Token did not contain email"}), 400

        # Check if user exists, if not create one
        user = users_collection.find_one({"email": email})
        
        if not user:
            # Create a new user account without a password (since they use Google)
            user_doc = {
                "name": name,
                "email": email,
                "auth_provider": "google",
                "created_at": datetime.utcnow()
            }
            result = users_collection.insert_one(user_doc)
            user_id = str(result.inserted_id)
        else:
            user_id = str(user["_id"])
            # Update name if missing
            if "name" not in user or not user["name"]:
                users_collection.update_one({"_id": user["_id"]}, {"$set": {"name": name}})
            
        # Generate JWT token
        payload = {
            "user_id": user_id,
            "email": email,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        jwt_token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

        response = make_response(jsonify({
            "message": "Login successful",
            "user": {
                "id": user_id,
                "name": name,
                "email": email
            }
        }), 200)

        response.set_cookie(
            "token",
            jwt_token,
            httponly=True,
            samesite="Lax",
            max_age=86400
        )
        return response

    except ValueError:
        return jsonify({"error": "invalid_token", "message": "Invalid Google token"}), 401
    except Exception as e:
        current_app.logger.exception("Error during Google login")
        return jsonify({"error": "server_error", "message": str(e)}), 500


@app.route("/api/logout", methods=["GET"])
def logout():
    """Clear the authentication cookie."""
    response = make_response(jsonify({
        "message": "Logged out successfully"
    }), 200)
    response.set_cookie("token", "", httponly=True, samesite="Lax", max_age=0)
    return response


@app.route("/api/detect", methods=["POST"])
@token_required
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
