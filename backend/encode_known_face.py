# update_known_faces.py
import os
import pickle
import face_recognition
import numpy as np

# ───────────────────────────────
# SETTINGS
# ───────────────────────────────
KNOWN_DIR  = "uploads"
CACHE_PATH = "data/alerts/known_faces.pkl"

# ───────────────────────────────
# 1) LOAD EXISTING CACHE
# ───────────────────────────────
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        known_files, known_names, known_encodings = pickle.load(f)
    known_files = set(known_files)
    print(f"🗄️  Loaded cache with {len(known_files)} files.")
else:
    known_files = set()
    known_names = []
    known_encodings = []
    print("🗄️  No cache found, starting fresh.")

# ───────────────────────────────
# 2) FIND NEW FILES
# ───────────────────────────────
all_files = [f for f in os.listdir(KNOWN_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

new_files = [f for f in all_files if f not in known_files]
if not new_files:
    print("✅ No new images found. Cache is up to date.")
    exit(0)

# ───────────────────────────────
# 3) ENCODE & APPEND NEW ONES
# ───────────────────────────────
for file in new_files:
    path = os.path.join(KNOWN_DIR, file)
    name = os.path.splitext(file)[0]

    img = face_recognition.load_image_file(path)
    encs = face_recognition.face_encodings(img)
    if not encs:
        print(f"⚠️  No face found in {file}, skipping.")
        continue

    vec = encs[0] / np.linalg.norm(encs[0])
    known_files.add(file)
    known_names.append(name)
    known_encodings.append(vec)
    print(f"✅ Encoded & added: {file} → `{name}`")

# ───────────────────────────────
# 4) SAVE UPDATED CACHE
# ───────────────────────────────
with open(CACHE_PATH, "wb") as f:
    # we pickle three parallel lists: filenames, labels, vectors
    pickle.dump((list(known_files), known_names, known_encodings), f)
print(f"💾 Cache updated. Now contains {len(known_files)} files.")
