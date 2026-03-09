from flask import Blueprint, request, jsonify
import bcrypt
from middleware.auth_middleware import token_required
from models.user_model import get_user_by_id, update_user_password

security_bp = Blueprint("security", __name__)

@security_bp.route("/api/security/change-password", methods=["POST"])
@token_required
def change_password():
    data = request.get_json(silent=True) or {}
    current_password = data.get("current_password")
    new_password = data.get("new_password")
    
    if not current_password or not new_password:
        return jsonify({"error": "missing_fields", "message": "Both current and new passwords are required"}), 400
        
    if len(new_password) < 8:
        return jsonify({"error": "invalid_password", "message": "New password must be at least 8 characters"}), 400
        
    user = get_user_by_id(request.user["user_id"])
    if "password" not in user or not user["password"]:
        return jsonify({"error": "no_password", "message": "User does not have a local password (Google Login)"}), 400
        
    if not bcrypt.checkpw(current_password.encode('utf-8'), user["password"]):
        return jsonify({"error": "incorrect_password", "message": "Current password is incorrect"}), 400
        
    if update_user_password(request.user["user_id"], new_password):
        return jsonify({"message": "Password updated successfully"}), 200
    return jsonify({"error": "server_error", "message": "Could not update password"}), 500
