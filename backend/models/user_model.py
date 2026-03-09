from bson import ObjectId
import bcrypt
from config.db import users_collection

def get_user_by_id(user_id):
    if isinstance(user_id, str):
        try:
            user_id = ObjectId(user_id)
        except:
            return None
    return users_collection.find_one({"_id": user_id})

def update_user_profile(user_id, name, profile_image=None):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    update_data = {"name": name}
    if profile_image is not None:
        update_data["profile_image"] = profile_image
        
    result = users_collection.update_one(
        {"_id": user_id},
        {"$set": update_data}
    )
    return result.modified_count > 0

def update_user_password(user_id, new_password):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
    
    result = users_collection.update_one(
        {"_id": user_id},
        {"$set": {"password": hashed}}
    )
    return result.modified_count > 0

def enable_user_two_factor(user_id, secret):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    result = users_collection.update_one(
        {"_id": user_id},
        {"$set": {"two_factor_enabled": True, "two_factor_secret": secret}}
    )
    return result.modified_count > 0

def disable_user_two_factor(user_id):
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
        
    result = users_collection.update_one(
        {"_id": user_id},
        {"$set": {"two_factor_enabled": False}, "$unset": {"two_factor_secret": ""}}
    )
    return result.modified_count > 0
