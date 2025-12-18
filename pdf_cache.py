import os
import json

CACHE_DIR = "data/processed"
META_FILE = os.path.join(CACHE_DIR, "meta.json")


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def load_metadata():
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_metadata(meta):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def save_text(doc_id, text):
    path = os.path.join(CACHE_DIR, f"doc_{doc_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def load_text(doc_id):
    path = os.path.join(CACHE_DIR, f"doc_{doc_id}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
