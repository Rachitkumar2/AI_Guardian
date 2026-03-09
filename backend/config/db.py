import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")

client = MongoClient(MONGO_URI)
db = client["ai_guardian"]
users_collection = db["users"]

# Create unique index on email
users_collection.create_index("email", unique=True)
