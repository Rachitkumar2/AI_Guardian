from config.db import detections_collection

def clear_scans():
    # Clear all guest (unauthenticated) scan records
    result = detections_collection.delete_many({"user_id": None})
    print(f"Cleared {result.deleted_count} anonymous scan records.")

if __name__ == "__main__":
    clear_scans()
