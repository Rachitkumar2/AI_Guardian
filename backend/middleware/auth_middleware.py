import os
import jwt
from flask import request, jsonify
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is not set")

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
