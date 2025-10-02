"""
FastAPI one-file backend for secure face registration
(keeps the *exact* thresholds & helper functions from your standalone script)
Python 3.9+, GPU optional
"""

from fastapi.staticfiles import StaticFiles
import csv, glob
from filelock import FileLock
import io


import os, ssl, smtplib, re, csv, uuid, pickle, shutil, time,uuid, aiofiles
from datetime import datetime
from typing import List, Optional, Tuple
import aiosqlite
from token_db import init_db, issue, check, DB

import cv2, numpy as np, torch, onnxruntime as ort
from PIL import Image
from tensorflow.keras.models import load_model
from timm import create_model
from torchvision import transforms
from ultralytics import YOLO
import mediapipe as mp
from email.message import EmailMessage
import ssl, smtplib

# --- NEW LOGIC IMPORT ---
import face_recognition
import imageio.v2 as imageio
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, APIRouter, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from auth_route import OneTimeTokenRoute
from token_db import issue, init_db

import secure_pickle
# ────────────────────────────────────────────────────────────
# 0.  CONSTANTS  ––––––  (identical to your old script)
# ────────────────────────────────────────────────────────────

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")   # 👈 default to Vite

KNOWN_DIR  = "uploads"               # where proof photos land
CACHE_PATH = "data/known_faces.pkl"  # your cache file

YOLO_MODEL_PATH      = "../models/yolov8n-face.pt"
LIVENESS_MODEL_PATH  = "../models/mobilenetv3_large_best.pth"
XCEPTION_CKPT_PATH  = "../models/xception41_finetuned_best.pth"

SAVE_DIR             = "tmp"
PROOF_PHOTO_DIR      = "uploads"
LOG_PATH             = "data/video_registration_log.csv"

VIDEO_DURATION = 4
FPS            = 20
FRAME_W = 1920; FRAME_H = 1080
POSES    = ["Center", "Left", "Right"]

YAW_RANGES       = {"Center": (-10, 10), "Left": (12, 60), "Right": (-60, -12)}
VALID_YAW_RATIO  = 0.70
SAMPLE_RATE_YAW  = 0.2

# Liveness Constants
IMG_SIZE_LIVE    = 224
CONF_THRESH_YOLO = 0.30
LIVENESS_THR     = 0.5
SPOOF_RATE_MAX   = 0.5
SAMPLE_RATE_LIVE = 0.1
BBOX_EXPANSION   = 2.7
REG_BBOX_EXPANSION = 1.2

# Deepfake Constants
IMG_SIZE_DF    = 224
DEEPFAKE_THR   = 0.50
SAMPLE_RATE_DF = 0.2
BBOX_EXPANSION_DF = 1.5

DEVICE = "cuda"

# Email (use env vars for safety)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT   = int(os.getenv("SMTP_PORT", 587))
SENDER_EMAIL= "xxxxxxxxxxxxxxg@gmail.com"
SMTP_PASS   = "xxxxxxxxxxxxxxxxxxx"  # 16-digit Gmail/Outlook app pw
# ────────────────────────────────────────────────────────────
# 1.  APP BOOT
# ────────────────────────────────────────────────────────────
app = FastAPI(title="Face-Registration API")
protected = APIRouter(route_class=OneTimeTokenRoute)# guarded
import logging, traceback


# catch-all logger (keeps traceback even if middleware fails)
@app.exception_handler(Exception)
async def catch_all(request, exc):
    logging.error("UNCAUGHT EXCEPTION", exc_info=exc)
    raise exc


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[BOOT] Loading models …")
yolo = YOLO(YOLO_MODEL_PATH)

MOBILENET_PATH  = LIVENESS_MODEL_PATH

live_model = create_model("mobilenetv3_large_100", pretrained=False, num_classes=1)
live_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE))
live_model.to(DEVICE).eval()

transform_live = transforms.Compose([
    transforms.Resize((224, 224)),  # IMG_SIZE_LIVE = 224
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


df_model = create_model("xception41", pretrained=False, num_classes=1)
df_model.load_state_dict(torch.load(XCEPTION_CKPT_PATH, map_location=DEVICE))
df_model.to(DEVICE).eval()

df_tf = transforms.Compose([
    transforms.Resize((224, 224)),  # IMG_SIZE_DF = 224
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
MODEL_POINTS = np.array([
    (0.,0.,0.), (0.,-330.,-65.),
    (-225.,170.,-135.), (225.,170.,-135.),
    (-150.,-150.,-125.), (150.,-150.,-125.)
])


_email_re = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

cache = {
    "files":      [],   # [filename1, filename2, …]
    "metadata":   {},   # filename → {"name":…, "email":…}
    "embeddings": []    # [vec1, vec2, …] same order as files
}

# ────────────────────────────────────────────────────────────
# 2.  UTILITIES  – identical to your old helpers
# ────────────────────────────────────────────────────────────
def _send_mail(to_addr: str, subject: str, body: str):
    if not _email_re.fullmatch(to_addr):
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = to_addr
    msg.set_content(body, subtype="plain", charset="utf-8")

    ctx = ssl.create_default_context()
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls(context=ctx)
        s.login(SENDER_EMAIL, SMTP_PASS)
        s.send_message(msg)

def load_known_faces_cache():
    global cache
    cache_path = CACHE_PATH
    if not os.path.exists(cache_path):
        print("🗄️  No cache found; starting empty")
        cache = {"files": [], "metadata": {}, "embeddings": []}
        return

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    # old format: (files, names, vecs)
    if isinstance(data, tuple) and len(data) == 3:
        files, names, vecs = data
        print(f"🗄️  (Old cache) migrating {len(files)} entries → new format")
        cache = {"files": [], "metadata": {}, "embeddings": []}
        for fn, name, vec in zip(files, names, vecs):
            cache["files"].append(fn)
            cache["embeddings"].append(vec)
            cache["metadata"][fn] = {"name": name, "email": ""}
        # overwrite with new format
        with open(cache_path, "wb") as out:
            pickle.dump(cache, out)
        print("💾  Migration complete")
        return

    # new format?
    if isinstance(data, dict) and {"files","metadata","embeddings"} <= set(data):
        cache.update(data)
        print(f"🗄️  Loaded {len(cache['files'])} faces from cache")
        return

    raise RuntimeError(f"Unrecognized cache at {cache_path}")


def update_known_faces_cache(new_meta: dict[str, dict[str, str]]):
    global cache
    changed = False

    for uid, info in new_meta.items():
        if uid in cache["files"]:
            continue                      # already encoded

        path = os.path.join(KNOWN_DIR, f"{uid}.jpg.enc")   # ← derive filename
        if not os.path.isfile(path):
            print(f"⚠️  Proof not found: {path}")
            continue

        # decrypt → load image  (secure_pickle handles .enc)
        img_data = secure_pickle.read_dec(path)
        img = face_recognition.load_image_file(io.BytesIO(img_data))

        locs = face_recognition.face_locations(img) or [(0, img.shape[1], img.shape[0], 0)]
        encs = face_recognition.face_encodings(img, locs, model="small")
        if not encs:
            print(f"⚠️  No encoding for {uid}, skipping");  continue

        vec = encs[0] / np.linalg.norm(encs[0])

        cache["files"].append(uid)              # 👈  bare UID
        cache["embeddings"].append(vec)
        cache["metadata"][uid] = {              # 👈  keyed by UID
            "name":   info["name"],
            "email":  info["email"],
            "status": info.get("status", "allowed"),
        }
        changed = True
        print(f"✅ Cached {uid} → {info['name']}")

    if changed:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        print(f"💾 Cache updated; total {len(cache['files'])} faces")



def expand_bbox(x1,y1,x2,y2,shape,scale=BBOX_EXPANSION):
    w,h = x2-x1, y2-y1; cx,cy = x1+w//2, y1+h//2
    nw,nh=int(w*scale),int(h*scale)
    return (max(0,cx-nw//2),max(0,cy-nh//2),
            min(shape[1],cx+nw//2),min(shape[0],cy+nh//2))

def crop_video_to_portrait(input_path, output_path, width=1280, height=1280):
    target_aspect = width / height
    cap = cv2.VideoCapture(str(input_path))
    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # webm-compatible
    out = cv2.VideoWriter(str(output_path), fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        input_aspect = w / h


        if abs(input_aspect - target_aspect) < 0.01:
            # Already close enough to 9:16, no need to crop
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_CUBIC)

        elif input_aspect > target_aspect:
            # Too wide — crop horizontally
            new_width = int(h * target_aspect)
            x_start = (w - new_width) // 2
            cropped = frame[:, x_start:x_start + new_width]
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_CUBIC)

        else:
            # Too tall — crop vertically
            new_height = int(w / target_aspect)
            y_start = (h - new_height) // 2
            cropped = frame[y_start:y_start + new_height, :]
            resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA if cropped.shape[1] > width else cv2.INTER_CUBIC)


        out.write(resized)

    cap.release()
    out.release()



def _yaw(img, coords):
    # coords is a list of (x, y) pixels for **all** 468 landmarks
    h, w = img.shape[:2]
    image_points = np.array([
        coords[1],   # nose tip
        coords[152], # chin
        coords[263], # left eye
        coords[33],  # right eye
        coords[287], # left mouth corner
        coords[57],  # right mouth corner
    ], dtype="double")

    cam = np.array([[w, 0, w / 2],
                    [0, w, h / 2],
                    [0, 0, 1]], dtype="double")

    ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points,
                                  cam, np.zeros((4, 1)))
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    proj = np.hstack([rot, tvec])
    _, _, _, _, _, _, eul = cv2.decomposeProjectionMatrix(proj)
    return eul[1][0]   # yaw in degrees


def _validate_yaw(video, pose):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, int(fps * SAMPLE_RATE_YAW))
    idx = tot = ok = 0

    while True:
        ret, f = cap.read(); idx += 1
        if not ret: break
        f = cv2.flip(f, 1)          # <— add this
        if idx % step: continue

        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks: continue

        lm = res.multi_face_landmarks[0].landmark
        h, w = f.shape[:2]
        coords = [(int(p.x * w), int(p.y * h)) for p in lm]   # ← NEW
        yaw = _yaw(f, coords)                                 # ← pass coords

        tot += 1
        if yaw and YAW_RANGES[pose][0] <= yaw <= YAW_RANGES[pose][1]:
            ok += 1

    cap.release()
    return False if tot == 0 else ok / tot >= VALID_YAW_RATIO

# ────────────────── LIVENESS ──────────────────────────────
def _live_ok(video) -> Tuple[Optional[bool], float]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, int(fps * SAMPLE_RATE_LIVE))
    total = spoof = idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % step != 0:
            continue
        det = yolo.predict(frame, conf=CONF_THRESH_YOLO, device=DEVICE, verbose=False)[0]
        if not det.boxes:
            continue
        x1, y1, x2, y2 = det.boxes.xyxy.cpu().numpy().astype(int)[0]
        bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, frame.shape, scale=BBOX_EXPANSION)
        face = frame[by1:by2, bx1:bx2]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_tensor = transform_live(Image.fromarray(face_rgb)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = live_model(face_tensor)
            p_real = torch.sigmoid(output).item()
        is_real = p_real <= LIVENESS_THR  # Real if ≤ threshold (same logic from your script)
        total += 1
        if not is_real:
            spoof += 1
    cap.release()
    spoof_rate = spoof / total if total else 1.0
    print(f"[LIVENESS] sampled={total}, spoof={spoof}, spoof_rate={spoof_rate:.2%}")
    passed = spoof_rate <= SPOOF_RATE_MAX
    return passed, spoof_rate



# ────────────────── DEEP-FAKE ─────────────────────────────
def _df_ok(video) -> Tuple[Optional[bool], float]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, int(fps * SAMPLE_RATE_DF))
    scores = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % step != 0:
            continue
        det = yolo.predict(frame, conf=CONF_THRESH_YOLO, device=DEVICE, verbose=False)[0]
        if not det.boxes:
            continue
        x1, y1, x2, y2 = det.boxes.xyxy.cpu().numpy().astype(int)[0]
        bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, frame.shape, scale=BBOX_EXPANSION_DF)
        face = frame[by1:by2, bx1:bx2]
        if face.size == 0:
            continue
        img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        t = df_tf(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.cuda.amp.autocast():
            prob_fake = torch.sigmoid(df_model(t)).item()
        scores.append(prob_fake)
    cap.release()
    if not scores:
        return None, 1.0  # No face found defaults to spoof/fake.
    mean_prob = sum(scores) / len(scores)
    print(f"[DEEPFAKE] sampled={len(scores)}, avg_fake_prob={mean_prob:.2%}")
    passed = mean_prob < DEEPFAKE_THR
    return passed, mean_prob



# ────────────────── OTHER HELPERS ────────────────────────────────────
def _save_proof(center_video: str, uid: str):
    import cv2, numpy as np, os
    from ultralytics import YOLO

    print(f"\n[DEBUG _save_proof] Extracting largest face frame for UID: {uid}")
    if not os.path.exists(center_video):
        print(f"[DEBUG _save_proof] ERROR: Video file does not exist.")
        return

    cap = cv2.VideoCapture(center_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    step = max(1, int(fps * 0.5))   # sample twice per second
    yolo = YOLO(YOLO_MODEL_PATH)

    best_face  = None
    max_area   = 0
    frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        det = yolo.predict(frame, conf=CONF_THRESH_YOLO, device=0, verbose=False)[0]
        if not det.boxes:
            continue

        h, w = frame.shape[:2]
        for box in det.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box

            # 1) Expand the box slightly for context
            bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, frame.shape, scale=REG_BBOX_EXPANSION)

            # 2) Add a small padding (5% of box size)
            bw, bh = x2 - x1, y2 - y1
            pad = int(0.05 * max(bw, bh))
            bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
            bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)

            area = (bx2 - bx1) * (by2 - by1)
            if area > max_area:
                candidate = frame[by1:by2, bx1:bx2]
                if candidate.size > 0:
                    best_face = candidate
                    max_area   = area

    cap.release()

    if best_face is not None:
        os.makedirs(PROOF_PHOTO_DIR, exist_ok=True)
        save_path = os.path.join(PROOF_PHOTO_DIR, f"{uid}.jpg.enc")
        secure_pickle.write_enc(save_path, cv2.imencode(".jpg", best_face)[1].tobytes())
        print(f"[DEBUG _save_proof] ✅ Saved largest face (area={max_area}) → {save_path}")
    else:
        print("[DEBUG _save_proof] ❌ No valid face found for proof photo.")





def _log(uid: str,
         name: str,
         email: str,
         live_pass: bool,
         df_pass: bool,
         spoof_rates: dict[str, float],
         df_rates:    dict[str, float],
         extra: Optional[str] = None):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ensure log directory & header
    hdr_needed = not os.path.exists(LOG_PATH)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if hdr_needed:
            w.writerow([
                "UserID", "Name", "Email",
                "LivePass", "DeepfakePass",
                "Spoof_Center", "Spoof_Left", "Spoof_Right",
                "DF_Center",    "DF_Left",    "DF_Right",
                "Timestamp"
            ])

        # helper to pull or default
        def pick(d, key):
            return f"{d[key]:.3f}" if key in d else "N/A"

        row = [
            uid, name, email,
            str(live_pass), str(df_pass),
            pick(spoof_rates, "Center"),
            pick(spoof_rates, "Left"),
            pick(spoof_rates, "Right"),
            pick(df_rates, "Center"),
            pick(df_rates, "Left"),
            pick(df_rates, "Right"),
            ts
        ]
        w.writerow(row)

# ────────────────────────────────────────────────────────────
# 3.  ENDPOINTS
# ────────────────────────────────────────────────────────────
from fastapi import FastAPI
import traceback, sys, ssl, smtplib
from token_db import init_db
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from timm import create_model
import onnxruntime as ort
import tensorflow as tf


# ------------------------------------------------------------------ #
#  Startup event – runs once when Uvicorn spins up
# ------------------------------------------------------------------ #

app.mount("/static", StaticFiles(directory="."), name="static")

@app.on_event("startup")
async def boot():
    await init_db()
    # The global model loading is handled at the top level now
    load_known_faces_cache()
    print("[BOOT] Models loaded OK ✓")


@app.post("/invite")
async def invite(email: str):
    tok  = await issue(email)
    link = f"{FRONTEND_URL}/?token={tok}"
    _send_mail(email, "One-time link", f"This is Chan Wen Hung FYP AI-Doorbell System Invitation Link."
                                       f"\n\nClick once:\n{link}\n\n"
                                       f"Regards,\nSecurity Team\n")
    return {"token": tok, "link": link}

@protected.get("/validate_token")
async def validate_token(token: str):
    """
    Returns the email associated with this one-time token,
    without marking it as used.
    """
    # first check that it exists and hasn’t been consumed
    is_valid = await check(token)
    if not is_valid:
        raise HTTPException(403, "Invalid or already-used token")

    # if you need the email in your frontend:
    async with aiosqlite.connect(str(DB)) as db:
        cur = await db.execute("SELECT email FROM tokens WHERE token=?", (token,))
        row = await cur.fetchone()

    return {"email": row[0]}


@protected.post("/register")
async def register(
    background: BackgroundTasks,
    name : str = Form(...),
    email: str = Form(...),
    center: UploadFile = File(...),
    left  : UploadFile = File(...),
    right : UploadFile = File(...),
):
    """Accepts three webm blobs named center / left / right and stores them encrypted."""
    if not _email_re.fullmatch(email):
        raise HTTPException(400, "Bad email")

    uid    = uuid.uuid4().hex
    folder = os.path.join(SAVE_DIR, uid)
    os.makedirs(folder, exist_ok=True)

    # -------- save & encrypt each upload immediately -----------------
    for pose, file in zip(POSES, (center, left, right)):
        enc_path = os.path.join(folder, f"{pose}.webm.enc")

        # 1️⃣ read browser bytes, encrypt immediately
        raw_bytes = await file.read()
        secure_pickle.write_enc(enc_path, raw_bytes)

        # 2️⃣ decrypt ▶ crop ▶ re‑encrypt
        tmp_plain = secure_pickle.read_dec(enc_path)

        # ---- crop (to 1000×1280) in memory ----
        tmp_in = os.path.join(folder, f"{pose}_plain.webm")
        tmp_out = os.path.join(folder, f"{pose}_cropped.webm")

        with open(tmp_in, "wb") as f:
            f.write(tmp_plain)

        crop_video_to_portrait(tmp_in, tmp_out, width=1000, height=1280)
        os.remove(tmp_in)

        with open(tmp_out, "rb") as f:
            cropped_bytes = f.read()
        os.remove(tmp_out)

        # overwrite .enc with the cropped video
        secure_pickle.write_enc(enc_path, cropped_bytes)

    # -----------------------------------------------------------------
    background.add_task(_process_job, uid, name, email)
    return JSONResponse({"uid": uid, "status": "processing"})




POSE_NAMES = ["Center", "Left", "Right"]
POSE_SET = set(POSE_NAMES)

@protected.post("/validate_yaw/{pose}")
async def validate_yaw_endpoint(
    pose: str,
    video: UploadFile = File(...),
):
    print("[debug] pose =", pose)          # ① path parameter

    if pose not in POSES:
        raise HTTPException(400, "pose must be Center | Left | Right")

    tmp_path: Optional[str] = None         # will hold the temp filename
    try:
        # ── read upload ─────────────────────────────────────────────
        data = await video.read()
        print("[debug] upload size =", len(data))      # ② bytes received
        if not data:
            raise HTTPException(400, "Empty upload")

        # ── save to ./tmp as .webm ──────────────────────────────────
        os.makedirs("tmp", exist_ok=True)
        enc_path = f"tmp/{uuid.uuid4().hex}_{pose}.webm.enc"
        async with aiofiles.open(enc_path, "wb") as out:
            await out.write(secure_pickle.encrypt_bytes(data))
        print("[debug] saved to", enc_path)

        # decrypt to temp plain file for OpenCV
        tmp_path = enc_path.replace(".webm.enc", "_dec.webm")
        with open(tmp_path, "wb") as t:
            t.write(secure_pickle.read_dec(enc_path))
        # ── heavy yaw-validation ───────────────────────────────────
        try:
            ok = _validate_yaw(tmp_path, pose)
            print("[debug] yaw-pass =", ok)  # add this line
            os.remove(tmp_path)  # remove plaintext temp
            os.remove(enc_path)  # (optional) tmp cleanup
            return JSONResponse({"pass": bool(ok)})   # always valid JSON
        except Exception as inner:
            print("[validate_yaw] _validate_yaw crashed:")
            import traceback; traceback.print_exc()
            raise HTTPException(400, "Could not read video: " + str(inner))


    except HTTPException:
        raise                                           # re-raise custom 400s
    except Exception as e:
        import traceback; traceback.print_exc()         # log everything else
        raise HTTPException(400, str(e))                # convert to 400
    finally:

        for p in (tmp_path, enc_path):  # clean up both files
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

app.include_router(protected)     # 👈 mount the protected router

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/archive", StaticFiles(directory="archive"), name="archive")



@app.get("/status/{uid}")
def status(uid: str):
    """Very naive status checker (exists = done)."""
    if os.path.exists(os.path.join(PROOF_PHOTO_DIR,f"{uid}.jpg.enc")):
        return {"status":"done"}
    if os.path.exists(os.path.join(SAVE_DIR,uid)):
        return {"status":"processing"}
    return {"status":"done"}


# ------------------------------------------------------------------
# Admin: manual approval from dashboard
# ------------------------------------------------------------------
@app.post("/admin/approve_registration")
def admin_approve_registration(uid: str):
    candidates = glob.glob(os.path.join("archive", "*", f"FAIL_{uid}"))
    if not candidates:
        raise HTTPException(404, f"FAIL folder for {uid} not found")
    folder   = candidates[0]
    date_dir = os.path.dirname(folder)

    # -------- read CSV for name/email --------
    name = email = "N/A"
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["UserID"] == uid:
                    name, email = row["Name"], row["Email"]
                    break

    # -------- proof + embeddings -------------
    enc_center_video = os.path.join(folder, "Center.webm.enc")
    tmp_center = enc_center_video.replace(".webm.enc", "_dec.webm")
    with open(tmp_center, "wb") as t:
        t.write(secure_pickle.read_dec(enc_center_video))
    _save_proof(tmp_center, uid)
    os.remove(tmp_center)

    # 2) register its metadata + embedding
    update_known_faces_cache({uid: {"name": name, "email": email, "status": "allowed"}})

    # -------- rename FAIL_ → MANUAL_ ---------
    new_folder = os.path.join(date_dir, f"MANUAL_{uid}")
    try:
        if os.path.exists(new_folder):
            shutil.rmtree(new_folder)        # remove stale dir from retries
        shutil.move(folder, new_folder)      # works across drives
    except Exception as e:
        raise HTTPException(500, f"Rename failed: {e}")

    # -------- send success e-mail ------------
    email = email.strip()
    if email and _email_re.fullmatch(email):
        try:
            _send_mail(
                email,
                "Registration APPROVED",
                f"Hi {name or 'user'},\n\n"
                "An administrator has approved your registration. "
                "You can now use face-recognition services.\n\n"
                "Regards,\nSecurity Team\n"
            )
        except Exception as mail_err:
            logging.error("Mail send failed: %s", mail_err)

    return {"detail": f"{uid} promoted to MANUAL, embedding saved"}



# ------------------------------------------------------------------ #
#  Background worker — liveness + deep-fake only                     #
# ------------------------------------------------------------------ #
def _process_job(uid: str, name: str, email: str):
    spoof_rates: dict[str, float] = {}
    df_rates:    dict[str, float] = {}
    folder   = os.path.join(SAVE_DIR, uid)
    outcome  = "FAIL"          # assume failure until we finish all checks

    try:
        # ── iterate the three poses ───────────────────────────────
        for pose in POSE_NAMES:
            f = os.path.join(folder, f"{pose}.webm.enc")
            tmp = f.replace(".webm.enc", "_dec.webm")   # keep .webm extension
            with open(tmp, "wb") as t:
                t.write(secure_pickle.read_dec(f))
            # 1️⃣  Liveness
            passed, spoof_rate = _live_ok(tmp)
            spoof_rates[pose] = spoof_rate
            if passed is False:
                pct = spoof_rate * 100
                reason = (
                    f"Liveness failed – spoof rate {pct:.1f}% "
                    f"(limit {SPOOF_RATE_MAX*100:.0f}%)"
                )
                _send_mail(email, "Registration FAILED",
                           f"Hi {name},\n\nYour registration has failed. The reason is:\n{reason} in {pose} pose.\n\nRegards,\nSecurity Team")
                _log(uid, name, email,
                     live_pass=False,
                     df_pass="N/A",
                     spoof_rates=spoof_rates,
                     df_rates=df_rates,
                     extra=reason)
                return                          # early exit, outcome=FAIL

            # 2️⃣  Deep-fake
            passed, fake_prob = _df_ok(tmp)
            os.remove(tmp)  # tidy up
            df_rates[pose] = fake_prob
            if passed is False:
                pct = fake_prob * 100
                reason = (
                    f"Deep-fake detected – fake-prob {pct:.1f}% "
                    f"(thr {DEEPFAKE_THR*100:.0f}%)"
                )
                _send_mail(email, "Registration FAILED",
                           f"Hi {name},\n\nYour registration has failed. The reason is:\n{reason} in {pose} pose.\n\nRegards,\nSecurity Team")
                _log(uid, name, email,
                     live_pass=True,
                     df_pass=False,
                     spoof_rates=spoof_rates,
                     df_rates=df_rates,
                     extra=reason)
                return                          # early exit, outcome=FAIL

        enc_center = os.path.join(folder, "Center.webm.enc")
        tmp_center = enc_center.replace(".webm.enc", "_dec.webm")
        with open(tmp_center, "wb") as t:
            t.write(secure_pickle.read_dec(enc_center))
        _save_proof(tmp_center, uid)  # unchanged call
        os.remove(tmp_center)

        # --- CALL THE NEW REGISTRATION LOGIC ---
        update_known_faces_cache({uid: {"name": name, "email": email, "status": "allowed"}})

        _send_mail(email, "Registration SUCCESS",
                   f"Hi {name},\n\nYour face registration was successful!\n\nRegards,\nSecurity Team")
        _log(uid, name, email,
             live_pass=True,
             df_pass=True,
             spoof_rates=spoof_rates,
             df_rates=df_rates)

        outcome = "PASS"        # mark success

    except Exception as e:
        # unexpected crash – still archive folder
        print(f"[ERROR] Job {uid} failed:", e, flush=True)
        import traceback; traceback.print_exc()

    finally:
        # ── archive keep-everything policy ───────────────────────
        date = datetime.now().strftime("%Y%m%d")
        dst  = os.path.join("archive", date, f"{outcome}_{uid}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            try:
                shutil.move(folder, dst)
            except Exception as e:
                print("[archive] could not move folder:", e)

