from datetime import datetime
from bson import ObjectId
from config.db import sessions_collection, login_history_collection

def create_session(user_id, ip, browser, device):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    session_doc = {
        "user_id": user_id,
        "login_time": datetime.utcnow(),
        "ip": ip,
        "browser": browser,
        "device": device
    }
    result = sessions_collection.insert_one(session_doc)
    return str(result.inserted_id)

def get_active_sessions(user_id):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    sessions = list(sessions_collection.find({"user_id": user_id}).sort("login_time", -1))
    for s in sessions:
        s["_id"] = str(s["_id"])
        s["user_id"] = str(s["user_id"])
    return sessions

def delete_other_sessions(user_id, current_session_id):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    if isinstance(current_session_id, str) and current_session_id:
        current_session_id = ObjectId(current_session_id)
    else:
        # If no current_session_id provided, we might just delete all or it shouldn't happen
        current_session_id = None
        
    query = {"user_id": user_id}
    if current_session_id:
        query["_id"] = {"$ne": current_session_id}
        
    result = sessions_collection.delete_many(query)
    return result.deleted_count

def log_login_attempt(user_id, ip, browser, device, status):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    log_doc = {
        "user_id": user_id,
        "time": datetime.utcnow(),
        "ip": ip,
        "browser": browser,
        "device": device,
        "status": status  # "Success" or "Failed"
    }
    login_history_collection.insert_one(log_doc)

def get_login_history(user_id, limit=10):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    history = list(login_history_collection.find({"user_id": user_id}).sort("time", -1).limit(limit))
    for h in history:
        h["_id"] = str(h["_id"])
        h["user_id"] = str(h["user_id"])
    return history
