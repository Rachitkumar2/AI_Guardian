from datetime import datetime

from config.db import usage_collection

FREE_GUEST_SCAN_LIMIT = 2


def _normalize_user_id(user_id):
    if user_id is None:
        return None
    return str(user_id)


def _build_guest_query(guest_id=None, guest_ip=None):
    """Build a query that matches either the guest_id or guest_ip."""
    clauses = []
    if guest_id:
        clauses.append({"guest_id": guest_id})
    if guest_ip:
        clauses.append({"guest_ip": guest_ip})
    
    if not clauses:
        return None
    return {"$or": clauses}


def _fetch_usage(guest_id=None, user_id=None, guest_ip=None):
    if user_id:
        return usage_collection.find_one({"user_id": _normalize_user_id(user_id)})
    
    query = _build_guest_query(guest_id=guest_id, guest_ip=guest_ip)
    if not query:
        return None
    
    # We sort by scans_used descending to find the most restrictive record
    # (e.g. if one browser ID has 2 scans and another has 0, we take the 2).
    return usage_collection.find_one(query, sort=[("scans_used", -1)])


def allowed_to_scan(guest_id=None, user_id=None, guest_ip=None):
    normalized_user_id = _normalize_user_id(user_id)
    if normalized_user_id:
        # Authenticated users have their own limits/logic (currently unlimited in this code)
        record = _fetch_usage(user_id=normalized_user_id)
        return {
            "allowed": True,
            "limit": None,
            "scans_used": int(record.get("scans_used", 0)) if record else 0,
            "scans_remaining": None,
            "record": record,
        }

    record = _fetch_usage(guest_id=guest_id, guest_ip=guest_ip)
    scans_used = int(record.get("scans_used", 0)) if record else 0
    scans_remaining = max(0, FREE_GUEST_SCAN_LIMIT - scans_used)

    return {
        "allowed": scans_used < FREE_GUEST_SCAN_LIMIT,
        "limit": FREE_GUEST_SCAN_LIMIT,
        "scans_used": scans_used,
        "scans_remaining": scans_remaining,
        "record": record,
    }


def increment_usage(guest_id=None, user_id=None, guest_ip=None):
    normalized_user_id = _normalize_user_id(user_id)
    
    # 1. Determine the primary identity for this update
    if normalized_user_id:
        selector = {"user_id": normalized_user_id}
    elif guest_id:
        selector = {"guest_id": guest_id}
    elif guest_ip:
        selector = {"guest_ip": guest_ip}
    else:
        return None

    now = datetime.utcnow()
    update_doc = {
        "$inc": {"scans_used": 1},
        "$set": {"updated_at": now},
        "$setOnInsert": {"created_at": now},
    }

    # 2. Store all available identifiers in the record for future cross-referencing
    if normalized_user_id:
        update_doc["$set"]["user_id"] = normalized_user_id
    if guest_id:
        update_doc["$set"]["guest_id"] = guest_id
    if guest_ip:
        update_doc["$set"]["guest_ip"] = guest_ip

    # 3. Perform the update (upsert if no record exists for this specific selector)
    return usage_collection.update_one(selector, update_doc, upsert=True)


def merge_guest_usage_into_user(guest_id=None, user_id=None, guest_ip=None):
    normalized_user_id = _normalize_user_id(user_id)
    if not normalized_user_id:
        return None

    guest_record = _fetch_usage(guest_id=guest_id, guest_ip=guest_ip)
    if not guest_record:
        return None

    now = datetime.utcnow()
    guest_scans_used = int(guest_record.get("scans_used", 0))
    user_record = _fetch_usage(user_id=normalized_user_id)
    current_user_scans = int(user_record.get("scans_used", 0)) if user_record else 0

    usage_collection.update_one(
        {"user_id": normalized_user_id},
        {
            "$set": {
                "user_id": normalized_user_id,
                "scans_used": max(current_user_scans, guest_scans_used),
                "updated_at": now,
            },
            "$setOnInsert": {
                "created_at": now,
            },
        },
        upsert=True,
    )

    return usage_collection.find_one({"user_id": normalized_user_id})
