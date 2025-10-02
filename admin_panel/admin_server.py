import os
import glob
import csv
import pickle
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

import secure_pickle
from fastapi.responses import StreamingResponse, FileResponse
import mimetypes, io

import requests
from fastapi import Depends, FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
logger = logging.getLogger("admin.media")

# --- CONFIGURATION ---
# In a real app, use environment variables for these!
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
ADMIN_USERNAME = "admin"
# This is the hashed password for the string "password".
# To generate a new hash, run this file directly: `python admin_server.py`
ADMIN_PASSWORD_HASH = "$2b$12$1QuIoeIJbRxDiDV4twgNbuDZLwWFzqDbmfo11XSkSZGPoqTQReTeS"

# --- PATHS ---
ORIGINAL_BACKEND = "http://localhost:8003"
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

DB_PATH = PROJECT_ROOT / "backend/data/token_store.db"
REG_LOG_CSV = PROJECT_ROOT / "backend/data/video_registration_log.csv"
PROOF_DIR = PROJECT_ROOT / "backend/uploads"
ARCHIVE_BASE = PROJECT_ROOT / "backend/archive"
ATD_CSV = PROJECT_ROOT / "attendance_log.csv"
FRONTEND_DIR = SCRIPT_DIR / "admin_ui/dist"
CACHE_PATH = PROJECT_ROOT / "backend/data/known_faces.pkl"
# --- FASTAPI APP & SECURITY SETUP ---
app = FastAPI()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:8001"],  # Add both dev and prod origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- AUTHENTICATION UTILITIES ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_admin_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username != ADMIN_USERNAME:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return {"username": username}


# A type alias for the dependency for cleaner code
CurrentUser = Annotated[dict, Depends(get_current_admin_user)]


# --- API ENDPOINTS ---

@app.post("/api/token")
def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    if not (form_data.username == ADMIN_USERNAME and verify_password(form_data.password, ADMIN_PASSWORD_HASH)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/dashboard-stats")
def get_dashboard_stats(current_user: CurrentUser):
    try:
        proofs = list(PROOF_DIR.glob("*.jpg.enc"))  # ← changed extension
        total_users = len(proofs)
    except FileNotFoundError:
        total_users = 0

    try:
        con = sqlite3.connect(DB_PATH)
        total_invites = con.execute("SELECT COUNT(*) FROM tokens").fetchone()[0]
        con.close()
    except Exception:
        total_invites = 0

    attendance_today = 0
    recent_events = []
    try:
        with open(ATD_CSV, newline="", encoding='utf-8') as f:
            all_events = list(csv.DictReader(f))
            today = datetime.now().date()
            for event in all_events:
                try:
                    event_date = datetime.strptime(event.get('Timestamp', ''), "%Y-%m-%d %H:%M:%S").date()
                    if event_date == today:
                        attendance_today += 1
                except (ValueError, TypeError):
                    continue  # Skip rows with invalid date formats
            recent_events = all_events[-5:]  # Get last 5 events
    except FileNotFoundError:
        print(f"Warning: Attendance log file not found at {ATD_CSV}")

    return {
        "totalUsers": total_users,
        "totalInvites": total_invites,
        "attendanceToday": attendance_today,
        "recentEvents": recent_events
    }


@app.post("/api/invite")
def invite_user(email: str, current_user: CurrentUser):
    try:
        resp = requests.post(f"{ORIGINAL_BACKEND}/invite", params={"email": email})
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)


@app.get("/api/invites")
def invites(current_user: CurrentUser):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT token,email,used,issued FROM tokens").fetchall()
    con.close()
    return [{"token": t, "email": e, "used": bool(u), "issued": i} for t, e, u, i in rows]


@app.get("/api/registration-logs")
def get_registration_logs(current_user: CurrentUser):
    logs = []
    try:
        with open(REG_LOG_CSV, newline="", encoding='utf-8') as f:
            logs.extend(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Warning: Registration log file not found at {REG_LOG_CSV}")
        return []

    # --- NEW: Load the face cache to get user statuses ---
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        metadata = cache.get("metadata", {})
        # Create a simple lookup from uid -> status
        uid_to_status = {
            os.path.splitext(file_id)[0]: data.get("status", "allowed")
            for file_id, data in metadata.items()
        }
    except FileNotFoundError:
        uid_to_status = {}

    # --- NEW: Add the status to each log entry ---
    for log in logs:
        log_uid = log.get("UserID")
        log["status"] = uid_to_status.get(log_uid, "unknown")

    return logs


from pydantic import BaseModel


class StatusUpdateRequest(BaseModel):
    status: str


@app.post("/api/users/{uid}/status")
def update_user_status(uid: str, request: StatusUpdateRequest, current_user: CurrentUser):
    new_status = request.status
    if new_status not in ["allowed", "blocked"]:
        raise HTTPException(status_code=400, detail="Invalid status provided.")

    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Cache file not found.")

    metadata = cache.get("metadata", {})
    file_id_to_update = None

    # Find the file_id that corresponds to the uid
    for file_id in metadata:
        if file_id.startswith(uid):
            file_id_to_update = file_id
            break

    if not file_id_to_update:
        raise HTTPException(status_code=404, detail=f"User with UID {uid} not found in metadata.")

    # Update the status and save the file
    metadata[file_id_to_update]['status'] = new_status
    cache['metadata'] = metadata

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)

    return {"detail": f"Successfully updated user {uid} status to '{new_status}'."}

@app.get("/api/attendance-logs")
def get_attendance_logs(current_user: CurrentUser):
    logs = []
    try:
        with open(ATD_CSV, newline="", encoding='utf-8') as f:
            logs.extend(csv.DictReader(f))
    except FileNotFoundError:
        print(f"Warning: Attendance log file not found at {ATD_CSV}")
    return logs




@app.get("/api/media")
def media(current_user: CurrentUser):
    # List of proof-photo filenames
    proofs = [f"uploads/{p.name}" for p in sorted(PROOF_DIR.glob('*.jpg.enc'))]

    # Build archive rows
    archives = []
    for folder in glob.glob(str(ARCHIVE_BASE / "*" / "*")):
        p = Path(folder)

        # 1️⃣ must be a directory
        if not p.is_dir():
            logger.warning("Skipping non-dir in archive: %s", p)
            continue

        # 2️⃣ folder name must contain an underscore  →  STATUS_uid
        if "_" not in p.name:
            logger.warning("Skipping malformed archive folder: %s", p)
            continue

        status, uid = p.name.split("_", 1)
        date = p.parent.name

        files = [f.name for f in p.iterdir() if f.is_file()]
        archives.append(
            {
                "uid":    uid,
                "status": status.upper(),  # “PASS” / “FAIL”
                "date":   date,            # “YYYYMMDD”
                "files":  files,           # ["center.mp4", ...]
            }
        )

    return {"proofs": proofs, "archives": archives}

@app.get("/api/media/raw/{subpath:path}")
def serve_media(subpath: str):
    """
    Decrypt *.enc files on‑the‑fly so the browser can display them.
    Accepts paths like  uploads/abcd.jpg.enc  or  archive/20250719/PASS_123/Center.webm.enc
    """
    full = PROJECT_ROOT / "backend" / subpath
    if not full.exists():
        raise HTTPException(404, "File not found")

    # encrypted → decrypt → stream
    if full.suffix == ".enc":
        try:
            raw  = secure_pickle.read_dec(full)
            mime = (
                mimetypes.guess_type(full.with_suffix("").name)[0]
                or "video/webm"  # sensible default
            )
            return StreamingResponse(io.BytesIO(raw), media_type=mime)
        except Exception as e:
            raise HTTPException(500, f"Decrypt failed: {e}")

    # legacy plaintext fallback
    return FileResponse(full)

import csv
import pickle
from pathlib import Path

def _purge_user(uid: str):
    """
    Fully remove all traces of a user identified by `uid`:
      1) registration CSV row
      2) known_faces.pkl entries
      3) proof photos
      4) archive folders
    """
    # 1️⃣ CSV cleanup
    tmp_csv = REG_LOG_CSV.with_suffix(".tmp")
    with open(REG_LOG_CSV, newline="", encoding="utf-8") as src, \
         open(tmp_csv, "w", newline="", encoding="utf-8") as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst)
        header = next(reader, None)
        if header:
            writer.writerow(header)
        for row in reader:
            # assume UID is in first column
            if not row or row[0] == uid:
                continue
            writer.writerow(row)
    tmp_csv.replace(REG_LOG_CSV)
    print(f"[CLEANUP] Removed CSV entries for UID='{uid}'")

    # 2️⃣ Face‐cache (known_faces.pkl) cleanup
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)

            files      = cache.get("files", [])
            embeddings = cache.get("embeddings", [])
            metadata   = cache.get("metadata", {})

            # find filenames to delete
            to_delete = [fn for fn in files if fn.startswith(uid)]
            if to_delete:
                print(f"[CLEANUP] Deleting {len(to_delete)} embeddings: {to_delete}")

                # rebuild lists without those entries
                new_files      = []
                new_embeddings = []
                for fn, emb in zip(files, embeddings):
                    if fn not in to_delete:
                        new_files.append(fn)
                        new_embeddings.append(emb)

                # rebuild metadata
                new_metadata = {fn: meta for fn, meta in metadata.items()
                                if fn not in to_delete}

                # persist
                cache["files"]      = new_files
                cache["embeddings"] = new_embeddings
                cache["metadata"]   = new_metadata
                with open(CACHE_PATH, "wb") as f:
                    pickle.dump(cache, f)
                print(f"[CLEANUP] Removed {len(to_delete)} cache entries for UID='{uid}'")
            else:
                print(f"[CLEANUP] No cache entries found for UID='{uid}'")

        except Exception as e:
            print(f"[ERROR] Purging cache failed: {e}")

    # 3️⃣ Proof photos
    for proof in PROOF_DIR.glob(f"{uid}*.jpg.enc"):  # ← ext changed
        try:
            proof.unlink()
            print(f"[CLEANUP] Deleted proof photo: {proof.name}")
        except Exception as e:
            print(f"[ERROR] Could not delete proof photo {proof.name}: {e}")

    # 4️⃣ Archive folders
    for folder in ARCHIVE_BASE.glob(f"*/[!_]*_{uid}"):
        if folder.is_dir():
            try:
                for file in folder.iterdir():
                    file.unlink()
                folder.rmdir()
                print(f"[CLEANUP] Deleted archive folder: {folder.name}")
            except Exception as e:
                print(f"[ERROR] Could not delete archive folder {folder.name}: {e}")



@app.delete("/api/registration/{uid}")
def delete_registration(uid: str, current_user: CurrentUser, background: BackgroundTasks):
    background.add_task(_purge_user, uid)
    return {"detail": f"user {uid} scheduled for deletion"}

# ------------------------------------------------------------------
#  Admin: manual approval (dashboard → main backend)
# ------------------------------------------------------------------
@app.post("/api/registration/{uid}/approve")
def approve_registration(uid: str, current_user: CurrentUser):
    """
    • Mark ManualApproval in CSV
    • Call main backend /admin/approve_registration
    """
    # ---------- 1) forward to main backend ------------------------
    resp = requests.post(
        f"{ORIGINAL_BACKEND}/admin/approve_registration",
        params={"uid": uid},
        timeout=30,
    )

    try:
        resp.raise_for_status()  # 2xx → OK, fall through
    except requests.RequestException as e:
        # Connection error?  e.response is None  →  502 Bad Gateway
        status = getattr(e.response, "status_code", 502)
        raise HTTPException(status, f"Backend error: {e}") from e

    # ---------- 2) set ManualApproval=True -----------------------
    try:
        with open(REG_LOG_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        raise HTTPException(404, "Registration log not found")

          # ─── NEW: no data rows at all ────────────────────────────────

    if not rows:
        raise HTTPException(
    404,
        "Registration log has no entries; cannot mark ManualApproval.",
        )

    fns = rows[0].keys() if rows else ["UserID", "Name", "Email",
                                       "LivePass", "DeepfakePass",
                                       "Spoof_Center", "Spoof_Left",
                                        "Spoof_Right",
                                              "DF_Center", "DF_Left", "DF_Right",
                                           "Timestamp"]
    if "ManualApproval" not in fns:
        fns = list(fns) + ["ManualApproval"]
        for r in rows:
            r["ManualApproval"] = ""

    found = False
    for r in rows:
        if r["UserID"] == uid:
            r["ManualApproval"] = "True"
            found = True
            break
    if not found:
        raise HTTPException(404, f"User {uid} not in log")

    with open(REG_LOG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        w.writerows(rows)

    return {"detail": f"{uid} approved (manual)"}





#
# # # --- STATIC FILE SERVING ---
# app.mount("/uploads", StaticFiles(directory=PROOF_DIR), name="uploads")
# app.mount("/archive", StaticFiles(directory=ARCHIVE_BASE), name="archive")
# app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# --- STATIC FILE SERVING ---
# Mount the entire 'backend' directory. This correctly serves files from
# 'backend/uploads', 'backend/archive', and 'backend/data/alerts'.
app.mount("/backend", StaticFiles(directory=(PROJECT_ROOT / "backend")), name="backend")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# --- UTILITY TO RUN SERVER AND GENERATE HASHED PASSWORDS ---
if __name__ == "__main__":
    import uvicorn

    # # --- Use this to generate a new password hash if you need to ---
    # print("--- Password Hash Generator ---")
    # password_to_hash = "password" # Change this to your desired password
    # hashed_password = pwd_context.hash(password_to_hash)
    # print(f"Password: {password_to_hash}")
    # print(f"Hashed  : {hashed_password}")
    # print("-" * 30)

    print("Starting Admin Server...")
    uvicorn.run(app, host="127.0.0.1", port=8001)