import time
import requests
from generator import generate_features

API_URL = "http://localhost:8000/predict"

SEND_DRIFT = True   # 🔁 toggle this to False for baseline

INTERVAL_SECONDS = 2

while True:
    payload = {
        "features": generate_features(drift=SEND_DRIFT)
    }

    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        print("STATUS:", r.status_code, "RESPONSE:", r.json())
    except Exception as e:
        print("Request failed:", e)

    time.sleep(INTERVAL_SECONDS)
