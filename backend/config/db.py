import os
from pymongo import MongoClient
from pymongo.errors import ConfigurationError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

# Create client with sensible timeouts
client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=5000
)

# Try to use database from URI, fall back to ai_guardian
try:
    db = client.get_default_database()
except ConfigurationError:
    db = client["ai_guardian"]

users_collection = db["users"]
sessions_collection = db["sessions"]
login_history_collection = db["login_history"]
detections_collection = db["detections"]

# Create unique index on email
users_collection.create_index("email", unique=True)
