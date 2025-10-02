# secure_pickle.py  ← copy/paste this whole file
import os, pickle, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_KEY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "secret.key"))
_NONCE_LEN = 12             # standard nonce size for AES‑GCM

def _get_key() -> bytes:
    if os.path.exists(_KEY_FILE):
        return open(_KEY_FILE, "rb").read()
    key = AESGCM.generate_key(bit_length=256)
    with open(_KEY_FILE, "wb") as f:
        f.write(key)
    os.chmod(_KEY_FILE, 0o600)      # owner‑only permission (works on macOS/Linux/WSL)
    return key

_KEY = _get_key()
_aes = AESGCM(_KEY)

def _enc(b: bytes) -> bytes:
    nonce = os.urandom(_NONCE_LEN)
    return nonce + _aes.encrypt(nonce, b, None)

def _dec(b: bytes) -> bytes:
    nonce, ct = b[:_NONCE_LEN], b[_NONCE_LEN:]
    return _aes.decrypt(nonce, ct, None)

# --- monkey‑patch pickle everywhere ------------------------------------
_real_dump = pickle.dump
_real_load = pickle.load

def dump(obj, file, protocol=None):
    raw = pickle.dumps(obj, protocol=protocol)
    file.write(_enc(raw))

def load(file):
    blob = file.read()
    try:                       # first run: file might still be plaintext
        raw = _dec(blob)
    except Exception:
        raw = blob
    return pickle.loads(raw)

# ---- extra helpers for arbitrary binary blobs -------------------------
def encrypt_bytes(b: bytes) -> bytes:
    return _enc(b)

def decrypt_bytes(b: bytes) -> bytes:
    return _dec(b)

def write_enc(path: str, raw: bytes):
    with open(path, "wb") as f:
        f.write(_enc(raw))

def read_dec(path: str) -> bytes:
    with open(path, "rb") as f:
        return _dec(f.read())


pickle.dump = dump
pickle.load = load
