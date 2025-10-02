#!/usr/bin/env python3
"""
quick_cache_inspect.py

Inspect or interactively purge entries from the encrypted known‑faces cache.
• Works with secure_pickle (AES‑256‑GCM) – just import it first.
• Prints overall counts plus a per‑user summary.
• If run without --delete, it will prompt you to enter a UID to delete interactively.
"""

import sys, os, argparse, pickle
import secure_pickle        # ← monkey‑patches pickle.load/dump

# ----------------------------------------------------------------------
# 1) Argument parsing
# ----------------------------------------------------------------------
DEFAULT_PATH = os.path.join("backend", "data", "known_faces.pkl")
parser = argparse.ArgumentParser(
    description="Inspect or delete entries from encrypted face cache"
)
parser.add_argument(
    "cache",
    nargs="?",
    default=DEFAULT_PATH,
    help=f"path to cache file (default: {DEFAULT_PATH})"
)
parser.add_argument(
    "--delete",
    metavar="UID",
    help="remove all cache entries whose filenames start with this UID"
)
args = parser.parse_args()
path = args.cache

# ----------------------------------------------------------------------
# 2) Load and inspect
# ----------------------------------------------------------------------
if not os.path.isfile(path):
    sys.exit(f"❌ Cache not found: {path}")

with open(path, "rb") as f:
    data = pickle.load(f)

embeddings = data.get("embeddings", [])
files      = data.get("files", [])
metadata   = data.get("metadata", {})

# Print summary
print(f"✅  Embeddings : {len(embeddings):>4}")
print(f"📁  Files      : {len(files):>4}")
print(f"📄  Metadata   : {len(metadata):>4}\n")

# Function to perform deletion and persist

def perform_delete(uid: str):
    to_delete = [fn for fn in data['files'] if fn.startswith(uid)]
    if not to_delete:
        print(f"⚠️  No entries found for UID='{uid}'")
        return False
    print(f"🗑️  Deleting {len(to_delete)} entries for UID='{uid}': {to_delete}")
    # rebuild
    data['files']      = [fn for fn in data['files'] if not fn.startswith(uid)]
    data['embeddings'] = [emb for fn, emb in zip(files, embeddings) if not fn.startswith(uid)]
    data['metadata']   = {fn: meta for fn, meta in metadata.items() if not fn.startswith(uid)}
    # persist
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"💾 Cache updated; remaining entries: {len(data['files'])}")
    return True

# ----------------------------------------------------------------------
# 3) Immediate delete if --delete provided
# ----------------------------------------------------------------------
if args.delete:
    perform_delete(args.delete)
    sys.exit(0)

# ----------------------------------------------------------------------
# 4) Interactive mode
# ----------------------------------------------------------------------
if files:
    print("┌─ Cached users (UID | Name | Email) ─────────────────────────────────┐")
    for fn in files:
        uid   = os.path.splitext(fn)[0]
        info  = metadata.get(fn, {})
        name  = info.get("name",   "???")
        email = info.get("email",  "—")
        print(f"│ {uid:<32} | {name:<25} | {email:<25} │")
    print("└────────────────────────────────────────────────────────────────────┘")

    # Prompt user
    while True:
        choice = input("\nEnter UID to delete (or press Enter to exit): ").strip()
        if not choice:
            print("Exiting without changes.")
            break
        if perform_delete(choice):
            break
else:
    print("⚠️  No embeddings yet.")
