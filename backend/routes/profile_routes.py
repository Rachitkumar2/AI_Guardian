from flask import Blueprint, request, jsonify
from middleware.auth_middleware import token_required
from models.user_model import get_user_by_id, update_user_profile

profile_bp = Blueprint("profile", __name__)

@profile_bp.route("/api/profile", methods=["GET"])
@token_required
def get_profile():
    user = get_user_by_id(request.user["user_id"])
    if not user:
        return jsonify({"error": "not_found", "message": "User not found"}), 404
        
    return jsonify({
        "name": user.get("name", ""),
        "email": user.get("email", "")
    }), 200

@profile_bp.route("/api/profile", methods=["PUT"])
@token_required
def update_profile():
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    
    if not name:
        return jsonify({"error": "missing_fields", "message": "Name is required"}), 400
        
    # We still use update_user_profile but pass None for image
    success = update_user_profile(request.user["user_id"], name, None)
    if success or get_user_by_id(request.user["user_id"])["name"] == name:
        return jsonify({"message": "Profile updated successfully"}), 200
    return jsonify({"error": "update_failed", "message": "Failed to update profile"}), 500
