import subprocess
import sys
import time
import threading
import queue
from pathlib import Path

import psutil  # pip install psutil

# Directories
ROOT_DIR            = Path(__file__).parent
ADMIN_DIR           = ROOT_DIR / "admin_panel"
ADMIN_FRONTEND_DIR  = ADMIN_DIR   / "admin_ui"
BACKEND_DIR         = ROOT_DIR    / "backend"
FRONTEND_DIR        = ROOT_DIR    / "frontend"

# Services
services = [
    {"name":"Admin Server",       "cmd":[sys.executable,"-m","uvicorn","admin_server:app","--host","127.0.0.1","--port","8001","--reload"], "cwd":ADMIN_DIR,           "port":8001},
    {"name":"Main Backend",       "cmd":[sys.executable,"-m","uvicorn","main:app",        "--host","127.0.0.1","--port","8003","--reload"], "cwd":BACKEND_DIR,         "port":8003},
    {"name":"Frontend (React)",   "cmd":["npm","run","dev"],                                    "cwd":FRONTEND_DIR,        "port":5173},
    {"name":"Frontend Admin (UI)","cmd":["npm","run","dev"],                                    "cwd":ADMIN_FRONTEND_DIR,  "port":5174},
]

# Thread‑safe queue for user commands
cmd_queue = queue.Queue()

import psutil
import subprocess

def kill_proc_tree(proc: subprocess.Popen):
    """Kill proc and all of its children."""
    try:
        parent = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        return

    # kill children first
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    # then kill the parent itself
    try:
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def start_service(svc):
    """Launch svc['cmd'], returning a Popen whose entire tree we can kill reliably."""
    is_npm = svc["cmd"][0].lower() == "npm"
    cwd    = svc["cwd"]

    if is_npm:
        # build a single string so shell=True can pick up npm.cmd on Windows
        cmd_str = " ".join(svc["cmd"] + ["--", "--port", str(svc["port"])])
        proc = subprocess.Popen(
            cmd_str,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            svc["cmd"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )

    print(f"🚀 Started {svc['name']} (PID {proc.pid}) on port {svc['port']}")
    return proc



def stream_logs(proc, name):
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}")

def input_reader():
    while True:
        choice = input("Restart service (1-4), q=quit: ").strip().lower()
        cmd_queue.put(choice)
        if choice == 'q':
            break

if __name__=="__main__":
    # 1) Launch all services + log‐stream threads
    processes = []
    for svc in services:
        p = start_service(svc)
        processes.append(p)
        t = threading.Thread(target=stream_logs, args=(p, svc["name"]), daemon=True)
        t.start()

    # 2) Start input thread
    threading.Thread(target=input_reader, daemon=True).start()

    print("\n✅ All services launched!\n")

    # 3) Main loop: handle restart commands
    while True:
        cmd = cmd_queue.get()   # blocks until you type something
        if cmd == 'q':
            break
        if cmd in ('1','2','3','4'):
            idx = int(cmd)-1
            svc = services[idx]
            old = processes[idx]
            print(f"\n🔄 Restarting {svc['name']}...\n")
            kill_proc_tree(old)
            old.terminate()
            try:
                old.wait(timeout=5)
            except subprocess.TimeoutExpired:
                old.kill()
            time.sleep(1)
            new = start_service(svc)
            processes[idx] = new
            threading.Thread(target=stream_logs, args=(new, svc["name"]), daemon=True).start()
        else:
            print("❌ Invalid choice; enter 1–4 or q.")

    # 4) Shutdown
    print("\n🛑 Shutting down all services…")
    for p in processes:
        p.terminate()
    print("✅ Done.")
