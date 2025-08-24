from threading import RLock

api_data = {
    "behaviors": [],
    "products": {},
    "users": {},
    "sellers": {},
}

api_lock = RLock()
