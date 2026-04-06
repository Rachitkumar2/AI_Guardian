from config.db import usage_collection

def clear_scans():
    # Clear all guest (unauthenticated) usage records.
    result = usage_collection.delete_many({"user_id": None})
    print(f"Cleared {result.deleted_count} anonymous scan records.")

if __name__ == "__main__":
    clear_scans()
