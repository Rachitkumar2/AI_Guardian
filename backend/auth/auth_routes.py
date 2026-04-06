import os
import bcrypt
import requests
from datetime import datetime
from flask import Blueprint, request, jsonify, make_response, current_app
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from auth.jwt_utils import generate_token
from config.db import users_collection
from models.usage_model import merge_guest_usage_into_user
from models.session_model import create_session, log_login_attempt
from user_agents import parse
from dotenv import load_dotenv

load_dotenv()

auth_bp = Blueprint("auth", __name__)


def _extract_guest_id(data=None):
    if data and data.get("guest_id"):
        return data.get("guest_id")
    return request.headers.get("X-Guest-Id") or request.cookies.get("guest_id")

@auth_bp.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_request", "message": "Invalid or missing JSON body"}), 400

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
        guest_id = _extract_guest_id(data)

        try:
            merge_guest_usage_into_user(
                guest_id=guest_id,
                user_id=str(result.inserted_id),
                guest_ip=request.headers.get("X-Forwarded-For", request.remote_addr),
            )
        except Exception:
            current_app.logger.exception("Failed to merge guest usage into new account")

        # Generate JWT token
        token = generate_token({"_id": result.inserted_id, "email": email})

        # Set JWT in HTTP-only cookie and return user info
        response = make_response(jsonify({
            "message": "User created successfully",
            "token": token,
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
            secure=True,
            samesite="Strict",
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
            "message": "An internal server error occurred"
        }), 500


@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_request", "message": "Invalid or missing JSON body"}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({
            "error": "missing_fields",
            "message": "Email and password are required"
        }), 400

    try:
        user = users_collection.find_one({"email": email})
        guest_id = _extract_guest_id(data)
        guest_ip = request.headers.get("X-Forwarded-For", request.remote_addr)

        if not user:
            log_login_attempt(email, request.headers.get("X-Forwarded-For", request.remote_addr), request.headers.get("User-Agent", ""), "Unknown", "Failed")
            return jsonify({
                "error": "invalid_credentials",
                "message": "Invalid email or password"
            }), 401

        # Check if user has a local password (not OAuth-only)
        if "password" not in user or not user["password"]:
            return jsonify({
                "error": "no_local_password",
                "message": "Account uses Google sign-in; use OAuth login"
            }), 401

        if bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            # Record Session
            user_agent = parse(request.headers.get("User-Agent", ""))
            browser = f"{user_agent.browser.family} {user_agent.browser.version_string}"
            device = f"{user_agent.os.family} {user_agent.os.version_string}"
            ip = request.headers.get("X-Forwarded-For", request.remote_addr)
            
            session_id = create_session(user["_id"], ip, browser, device)
            log_login_attempt(user["_id"], ip, browser, device, "Success")

            try:
                merge_guest_usage_into_user(
                    guest_id=guest_id,
                    user_id=str(user["_id"]),
                    guest_ip=guest_ip,
                )
            except Exception:
                current_app.logger.exception("Failed to merge guest usage into login account")
            
            # Generate JWT token with session_id
            user_payload = dict(user)
            user_payload["session_id"] = session_id
            token = generate_token(user_payload)

            response = make_response(jsonify({
                "message": "Login successful",
                "token": token,
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
                secure=True,
                samesite="Strict",
                max_age=86400
            )
            return response
        else:
            log_login_attempt(user["_id"], request.headers.get("X-Forwarded-For", request.remote_addr), request.headers.get("User-Agent", ""), "Unknown", "Failed")
            return jsonify({
                "error": "invalid_credentials",
                "message": "Invalid email or password"
            }), 401

    except Exception as e:
        current_app.logger.exception("Error during login")
        return jsonify({
            "error": "server_error",
            "message": "An internal server error occurred"
        }), 500


@auth_bp.route("/api/google-login", methods=["POST"])
def google_login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "invalid_request", "message": "Invalid or missing JSON body"}), 400

    token = data.get("credential")
    access_token = data.get("access_token")
    
    if not token and not access_token:
        return jsonify({"error": "missing_token", "message": "Google token is required"}), 400

    try:
        email = None
        name = None
        guest_id = _extract_guest_id(data)
        
        if access_token:
            try:
                resp = requests.get(
                    f"https://www.googleapis.com/oauth2/v3/userinfo?access_token={access_token}",
                    timeout=5
                )
            except requests.exceptions.Timeout:
                return jsonify({"error": "gateway_timeout", "message": "Google API request timed out"}), 504
            except requests.exceptions.RequestException as req_err:
                current_app.logger.exception("Google API request failed")
                return jsonify({"error": "bad_gateway", "message": "Failed to verify Google token"}), 502
            
            if resp.status_code != 200:
                return jsonify({"error": "invalid_token", "message": "Invalid Google access token"}), 400
            user_info = resp.json()
            email = user_info.get("email")
            name = user_info.get("name")
        elif token:
            # Verify the Google token
            client_id = os.environ.get("GOOGLE_CLIENT_ID")
            idinfo = id_token.verify_oauth2_token(token, google_requests.Request(timeout=5), client_id)
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
            # Check if this user was originally a local account
            existing_provider = user.get("auth_provider")
            
            # If they don't have a provider but have a password, they are local
            # or if they explicitly have "local" provider
            is_local = (existing_provider != "google" and "password" in user) or existing_provider == "local"
            
            # If so, seamlessly link their Google account
            if is_local:
                # Add Google auth provider info but keep their local password intact
                users_collection.update_one(
                    {"_id": user["_id"]}, 
                    {"$set": {"auth_provider": "google"}}
                )
            
            user_id = str(user["_id"])
            # Update name if missing
            if "name" not in user or not user["name"]:
                users_collection.update_one({"_id": user["_id"]}, {"$set": {"name": name}})
            else:
                name = user.get("name")
            
        # Record Session
        user_agent = parse(request.headers.get("User-Agent", ""))
        browser = f"{user_agent.browser.family} {user_agent.browser.version_string}"
        device = f"{user_agent.os.family} {user_agent.os.version_string}"
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        
        session_id = create_session(user_id, ip, browser, device)
        log_login_attempt(user_id, ip, browser, device, "Success")

        try:
            merge_guest_usage_into_user(
                guest_id=guest_id,
                user_id=user_id,
                guest_ip=ip,
            )
        except Exception:
            current_app.logger.exception("Failed to merge guest usage into Google account")

        # Generate JWT token
        user_payload = {"_id": user_id, "email": email, "session_id": session_id}
        jwt_token = generate_token(user_payload)

        response = make_response(jsonify({
            "message": "Login successful",
            "token": jwt_token,
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
            secure=True,
            samesite="Strict",
            max_age=86400
        )
        return response

    except ValueError:
        return jsonify({"error": "invalid_token", "message": "Invalid Google token"}), 401
    except Exception as e:
        current_app.logger.exception("Error during Google login")
        return jsonify({"error": "server_error", "message": "An internal server error occurred"}), 500


@auth_bp.route("/api/logout", methods=["POST"])
def logout():
    """Clear the authentication cookie. Uses POST to prevent CSRF attacks."""
    response = make_response(jsonify({
        "message": "Logged out successfully"
    }), 200)
    response.set_cookie("token", "", httponly=True, secure=True, samesite="Strict", max_age=0)
    return response
