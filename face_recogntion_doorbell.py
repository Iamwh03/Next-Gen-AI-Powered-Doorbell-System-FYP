#!/usr/bin/env python3
"""
doorbell_gui.py

A "Smart Doorbell" GUI application using Tkinter.
Features:
- Blocklist/Watchlist functionality for recognized users.
- 5-second cooldown on security alert snapshots and logging.
- Bounding Box Sanity Check for ultimate stability.
- Live video feed and recognition status.
"""
import os, io,uuid
import pickle
import threading
import queue
import time
import csv
from collections import deque

import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO

import torch
from timm import create_model
from torchvision import transforms
from PIL import Image, ImageTk

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, Frame, Label, Button, Canvas

import secure_pickle

# (Settings and helper functions are unchanged)
CACHE_PATH = "backend/data/known_faces.pkl"
UPLOADS_DIR = "backend/uploads"
YOLO_MODEL_PATH = "models/yolov8n-face.pt"
ALERT_SNAPSHOT_DIR = "backend/data/alerts"


SIM_THRESHOLD = 0.945
MARGIN_THRESHOLD = 0.015


PROCESS_SCALE = 0.5
HISTORY_LEN = 5
N_LIVE_CONSEC = 5
N_DF_CONSEC = 5
ALERT_THRESHOLD = 5
ALERT_COOLDOWN = 5
NO_FACE_GRACE_PERIOD = 5
MIN_BBOX_AREA = 40 * 40
XCEPTION_CKPT_PATH = "models/xception41_finetuned_best.pth"
IMG_SIZE_DF = 224
DEEPFAKE_THR = 0.4
BBOX_EXPANSION_DF = 1.8
MOBILENET_PATH = "models/mobilenetv3_large_best.pth"
IMG_SIZE_LIVE = 224
LIVENESS_THR = 0.5
BBOX_EXPANSION_LIVE = 2.7
DEVICE = "cpu"
DOORBELL_DISPLAY = 3;
DOORBELL_COOLDOWN = 2  # Durations in seconds

df_tf = transforms.Compose(
    [transforms.Resize((IMG_SIZE_DF, IMG_SIZE_DF)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
live_tf = transforms.Compose([transforms.Resize((IMG_SIZE_LIVE, IMG_SIZE_LIVE)), transforms.ToTensor(),
                              transforms.Normalize([0.5] * 3, [0.5] * 3)])


def expand_bbox(x1, y1, x2, y2, shape, scale):
    w, h = x2 - x1, y2 - y1;
    cx, cy = x1 + w // 2, y1 + h // 2;
    nw, nh = int(w * scale), int(h * scale)
    return (max(0, cx - nw // 2), max(0, cy - nh // 2), min(shape[1], cx + nw // 2), min(shape[0], cy + nh // 2))


class ThreadedCam:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src);
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width);
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.queue = queue.Queue(maxsize=2);
        self.stopped = False;
        self.thread = threading.Thread(target=self._reader, daemon=True);
        self.thread.start()

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret: self.stop();break
            if not self.queue.full():
                self.queue.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive(): self.thread.join(timeout=1)
        self.cap.release()


class DoorbellApp:
    def __init__(self, window_title="Smart Doorbell System"):
        self.root = tk.Tk();
        self.root.title(window_title);
        self.root.configure(bg="#212121")
        self.font_bold, self.font_normal, self.font_small = tkfont.Font(family="Helvetica", size=12,
                                                                        weight="bold"), tkfont.Font(family="Helvetica",
                                                                                                    size=10), tkfont.Font(
            family="Helvetica", size=9)
        self.colors = {"bg": "#212121", "fg": "#FAFAFA", "frame_bg": "#323232", "safe": "#4CAF50", "warn": "#FFC107",
                       "danger": "#F44336"}
        self.system_active = False;
        self.history = [];
        self.live_counter, self.df_counter = 0, 0;
        self.failed_live_counter, self.failed_df_counter = 0, 0
        self.no_face_counter = 0;
        self.recognized_time, self.recognized_name = None, None
        self.last_confidence = 0.0;
        self.bbox_history = deque(maxlen=5);
        self.yolo_model, self.df_model, self.live_model = None, None, None
        self.known_encs, self.metadata, self.files = [], {}, [];
        self.name_to_uid_map, self.uid_to_metadata_map = {}, {}
        self.proof_photo_image, self.last_displayed_uid = None, None
        self.last_liveness_alert_time = 0;
        self.last_deepfake_alert_time = 0
        self.last_blocklist_alert_time = 0
        os.makedirs(ALERT_SNAPSHOT_DIR, exist_ok=True)
        self.create_widgets();
        self.update_timestamp();
        self.cam = ThreadedCam(src=0)
        self.update_delay = 15;
        self.update_frame();
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing);
        self.root.mainloop()

    # create_widgets, save_alert_snapshot, etc. are unchanged
    def create_widgets(self):
        main_frame = Frame(self.root, bg=self.colors["bg"]);
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        left_frame = Frame(main_frame, bg=self.colors["frame_bg"]);
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        right_frame = Frame(main_frame, bg=self.colors["frame_bg"], width=250);
        right_frame.pack(side=tk.RIGHT, fill=tk.Y);
        right_frame.pack_propagate(False)
        self.canvas = Canvas(left_frame, width=640, height=480, bg="black", highlightthickness=0);
        self.canvas.pack(pady=5, padx=5)
        self.status_label = Label(left_frame, text="System Offline", font=self.font_bold, bg=self.colors["warn"],
                                  fg="black");
        self.status_label.pack(fill=tk.X, ipady=5, padx=5, pady=(0, 5))
        controls_frame = Frame(right_frame, bg=self.colors["frame_bg"]);
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        self.toggle_button = Button(controls_frame, text="Start System", command=self.toggle_system,
                                    font=self.font_normal, bg="#4CAF50", fg="white", activebackground="#45a049");
        self.toggle_button.pack(fill=tk.X, expand=True, padx=5)
        photo_title_label = Label(right_frame, text="Last Recognized", font=self.font_bold, bg=self.colors["frame_bg"],
                                  fg=self.colors["fg"]);
        photo_title_label.pack(pady=(10, 5))
        photo_container = Frame(right_frame, width=240, height=180, bg="#111111");
        photo_container.pack(pady=5, padx=5);
        photo_container.pack_propagate(False)
        self.proof_photo_label = Label(photo_container, bg="#111111");
        self.proof_photo_label.pack(fill=tk.BOTH, expand=True)
        info_frame = Frame(right_frame, bg=self.colors["frame_bg"]);
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self.email_label = Label(info_frame, text="Email: N/A", font=self.font_small, bg=self.colors["frame_bg"],
                                 fg=self.colors["fg"], anchor='w');
        self.email_label.pack(fill=tk.X)
        self.timestamp_label = Label(info_frame, text="Time: ---", font=self.font_small, bg=self.colors["frame_bg"],
                                     fg=self.colors["fg"], anchor='w');
        self.timestamp_label.pack(fill=tk.X, pady=(5, 0))

    def save_alert_snapshot(self, frame, alert_type):
        """
        Write   backend/data/alerts/{TYPE}_{timestamp}.jpg.enc
        and return the relative path used in attendance_log.
        """
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        uniq = uuid.uuid4().hex[:6]  # avoids name clashes in same second
        fname = f"{alert_type}_{ts}_{uniq}.jpg.enc"
        path = os.path.join(ALERT_SNAPSHOT_DIR, fname).replace("\\", "/")

        ok, buf = cv2.imencode(".jpg", frame)

        if not ok:
            print("[ALERT] JPEG encode failed");
            return "N/A"

        secure_pickle.write_enc(path, buf.tobytes())
        print(f"[ALERT] snapshot saved (encrypted) → {path}")
        return path

    def log_event(self, log_type, data=None):
        if data is None: data = {}
        log_file = 'attendance_log.csv'
        file_exists = os.path.isfile(log_file)
        with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['log_type', 'userid', 'name', 'confidence', 'timestamp', 'proofphotopath']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists: writer.writeheader()
            row = {'log_type': log_type, 'userid': data.get('userid', 'N/A'), 'name': data.get('name', 'N/A'),
                   'confidence': data.get('confidence', 'N/A'), 'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                   'proofphotopath': data.get('proof_path', 'N/A')}
            writer.writerow(row)

    def update_timestamp(self):
        now_str = time.strftime("%Y-%m-%d %I:%M:%S %p");self.timestamp_label.config(
            text=f"Time: {now_str}");self.root.after(1000, self.update_timestamp)

    def update_user_info(self, uid=None):
        if uid == self.last_displayed_uid:
            return

        # ---------- proof photo ----------
        photo_updated = False
        if uid:
            # first try encrypted file
            enc_path = os.path.join(UPLOADS_DIR, f"{uid}.jpg.enc")
            if os.path.exists(enc_path):
                try:
                    raw = secure_pickle.read_dec(enc_path)
                    img = Image.open(io.BytesIO(raw))
                    img.thumbnail((240, 180))
                    self.proof_photo_image = ImageTk.PhotoImage(img)
                    self.proof_photo_label.config(image=self.proof_photo_image)
                    photo_updated = True
                except Exception as e:
                    print(f"[GUI] could not decrypt/display {enc_path}: {e}")

            # fallback to legacy plaintext .jpg (optional)
            if (not photo_updated) and os.path.exists(os.path.join(UPLOADS_DIR, f"{uid}.jpg")):
                try:
                    img = Image.open(os.path.join(UPLOADS_DIR, f"{uid}.jpg"))
                    img.thumbnail((240, 180))
                    self.proof_photo_image = ImageTk.PhotoImage(img)
                    self.proof_photo_label.config(image=self.proof_photo_image)
                    photo_updated = True
                except Exception as e:
                    print(f"[GUI] could not open plaintext jpg: {e}")

        if not photo_updated:
            self.proof_photo_label.config(image='')

        # ---------- email ----------
        email = "N/A"
        if uid:
            user_data = self.uid_to_metadata_map.get(uid)
            if user_data:
                email = user_data.get('email', 'N/A')
        self.email_label.config(text=f"Email: {email}")

        self.last_displayed_uid = uid

    def setup_models_and_cache(self):
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.df_model = create_model("xception41", num_classes=1).to(DEVICE);
            self.df_model.load_state_dict(torch.load(XCEPTION_CKPT_PATH, map_location=DEVICE));
            self.df_model.eval()
            self.live_model = create_model("mobilenetv3_large_100", num_classes=1).to(DEVICE);
            self.live_model.load_state_dict(torch.load(MOBILENET_PATH, map_location=DEVICE));
            self.live_model.eval()
            if not os.path.exists(CACHE_PATH): messagebox.showerror("Error",
                                                                    f"Cache file not found: {CACHE_PATH}");return False
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            self.files, self.metadata = cache["files"], cache["metadata"]
            self.known_encs = [np.array(v) for v in cache["embeddings"]]
            self.name_to_uid_map = {d['name']: os.path.splitext(f)[0] for f, d in self.metadata.items()}
            self.uid_to_metadata_map = {os.path.splitext(f)[0]: d for f, d in self.metadata.items()}
            return True
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to load models/cache.\nError: {e}");return False

    def toggle_system(self):
        if not self.system_active:
            if self.setup_models_and_cache():
                self.system_active = True;
                self.toggle_button.config(text="Stop System", bg=self.colors["danger"], activebackground="#e53935");
                self.update_status("Scanning...", self.colors["safe"])
        else:
            self.system_active = False;
            self.toggle_button.config(text="Start System", bg=self.colors["safe"], activebackground="#45a049");
            self.update_status("System Offline", self.colors["warn"]);
            self.update_user_info(None)

    def update_status(self, text, color):
        self.status_label.config(text=text, bg=color)

    def process_frame(self, frame):
        # --- REFACTORED: Welcome message logic is now simpler and more reliable ---
        if self.recognized_time:
            elapsed = time.time() - self.recognized_time
            # Keep welcome message on screen for the full duration (display + cooldown)
            if elapsed < DOORBELL_DISPLAY + DOORBELL_COOLDOWN:
                self.update_status(f"Welcome, {self.recognized_name}!", self.colors["safe"])
                return frame
            else:
                # Timer is over, reset and continue to normal processing
                self.recognized_time, self.recognized_name = None, None
                self.history.clear()

        det = self.yolo_model.predict(frame, conf=0.4, device=DEVICE, verbose=False)[0]
        raw_bbox = None
        if len(det.boxes):
            potential_bbox = det.boxes.xyxy.cpu().numpy().astype(int)[0]
            if (potential_bbox[2] - potential_bbox[0]) * (potential_bbox[3] - potential_bbox[1]) > MIN_BBOX_AREA:
                raw_bbox = potential_bbox
        if raw_bbox is None:
            self.no_face_counter += 1
            if self.no_face_counter >= NO_FACE_GRACE_PERIOD:
                self.live_counter, self.df_counter, self.failed_live_counter, self.failed_df_counter = 0, 0, 0, 0
                self.update_user_info(None);
                self.bbox_history.clear()
            self.update_status("Scanning...", self.colors["safe"]);
            return frame

        self.no_face_counter = 0
        self.bbox_history.append(raw_bbox)
        avg_bbox = np.mean(self.bbox_history, axis=0).astype(int)
        x1, y1, x2, y2 = avg_bbox

        lx1, ly1, lx2, ly2 = expand_bbox(x1, y1, x2, y2, frame.shape, BBOX_EXPANSION_LIVE)
        crop_live = frame[ly1:ly2, lx1:lx2]
        is_real = crop_live.size > 0 and torch.sigmoid(self.live_model(
            live_tf(Image.fromarray(cv2.cvtColor(crop_live, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)))[
            0].item() <= LIVENESS_THR
        self.live_counter = self.live_counter + 1 if is_real else 0
        self.failed_live_counter = 0 if is_real else self.failed_live_counter + 1
        if self.failed_live_counter >= ALERT_THRESHOLD:
            if (time.time() - self.last_liveness_alert_time) > ALERT_COOLDOWN:
                snapshot_path = self.save_alert_snapshot(frame, 'LIVENESS_ALERT')
                self.log_event('LIVENESS_ALERT', data={'proof_path': snapshot_path})
                self.last_liveness_alert_time = time.time()
            self.failed_live_counter = 0
        if self.live_counter < N_LIVE_CONSEC:
            self.update_status(f"Verifying Liveness... ({self.live_counter}/{N_LIVE_CONSEC})", self.colors["warn"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 193, 7), 2);
            cv2.putText(frame, "VERIFYING...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 193, 7), 2)
            return frame

        bx1, by1, bx2, by2 = expand_bbox(x1, y1, x2, y2, frame.shape, BBOX_EXPANSION_DF)
        crop_df = frame[by1:by2, bx1:bx2]
        is_not_fake = crop_df.size > 0 and float(torch.sigmoid(self.df_model(
            df_tf(Image.fromarray(cv2.cvtColor(crop_df, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(
                DEVICE))).item()) < DEEPFAKE_THR
        self.df_counter = self.df_counter + 1 if is_not_fake else 0
        self.failed_df_counter = 0 if is_not_fake else self.failed_df_counter + 1
        if self.failed_df_counter >= ALERT_THRESHOLD:
            if (time.time() - self.last_deepfake_alert_time) > ALERT_COOLDOWN:
                snapshot_path = self.save_alert_snapshot(frame, 'DEEPFAKE_ALERT')
                self.log_event('DEEPFAKE_ALERT', data={'proof_path': snapshot_path})
                self.last_deepfake_alert_time = time.time()
            self.failed_df_counter = 0
        if self.df_counter < N_DF_CONSEC:
            self.update_status(f"Verifying Authenticity... ({self.df_counter}/{N_DF_CONSEC})", self.colors["warn"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 193, 7), 2);
            cv2.putText(frame, "VERIFYING...", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 193, 7), 2)
            return frame

        small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rects = [(int(y1 * PROCESS_SCALE), int(x2 * PROCESS_SCALE), int(y2 * PROCESS_SCALE), int(x1 * PROCESS_SCALE))]
        encs = face_recognition.face_encodings(rgb_small, rects, model="large")
        display_label = "Unknown"
        if encs:
            enc = encs[0] / np.linalg.norm(encs[0])
            sims = [float(np.dot(k, enc)) for k in self.known_encs]
            if sims:
                idxs = np.argsort(sims)[::-1]
                best, second = sims[idxs[0]], sims[idxs[1]] if len(idxs) > 1 else 0.0
                self.last_confidence = best
                if best >= SIM_THRESHOLD and (best - second) >= MARGIN_THRESHOLD:
                    display_label = self.metadata[self.files[idxs[0]]]["name"]

        self.history.append(display_label)
        if len(self.history) > HISTORY_LEN: self.history.pop(0)

        final_label = "Verifying..."
        box_color = (0, 0, 255)  # Default Red
        current_uid = None

        if len(self.history) == HISTORY_LEN and all(h == self.history[0] for h in self.history):
            final_label = self.history[0]
            if final_label not in ["Unknown", "Verifying..."]:
                current_uid = self.name_to_uid_map.get(final_label)
                user_data = self.uid_to_metadata_map.get(current_uid, {})
                status = user_data.get("status", "allowed")

                if status == 'blocked':
                    self.recognized_time = None
                    self.update_status(f"BLOCKLISTED USER: {final_label}", self.colors["danger"])
                    box_color = (128, 0, 128)
                    if (time.time() - self.last_blocklist_alert_time) > ALERT_COOLDOWN:
                        snapshot_path = self.save_alert_snapshot(frame, 'BLOCKLIST_ALERT')
                        log_data = {'userid': current_uid, 'name': final_label,
                                    'confidence': f"{self.last_confidence:.4f}", 'proof_path': snapshot_path}
                        self.log_event('BLOCKLIST_ALERT', data=log_data)
                        self.last_blocklist_alert_time = time.time()
                elif status == 'allowed':
                    box_color = (0, 255, 0)
                    if self.recognized_name is None:
                        # This is the single trigger for the welcome message and the log
                        self.recognized_name, self.recognized_time = final_label, time.time()
                        self.live_counter, self.df_counter = 0, 0
                        proof_path = os.path.join(UPLOADS_DIR, f"{current_uid}.jpg.enc").replace("\\", "/")
                        log_data = {'userid': current_uid, 'name': final_label,
                                    'confidence': f"{self.last_confidence:.4f}", 'proof_path': proof_path}
                        self.log_event('RECOGNITION', data=log_data)

        self.update_user_info(current_uid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        return frame

    def update_frame(self):
        frame = self.cam.read();
        frame = cv2.flip(frame, 1)
        if self.system_active: frame = self.process_frame(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
        img = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(self.update_delay, self.update_frame)

    def on_closing(self):
        self.cam.stop(); self.root.destroy()


if __name__ == "__main__":
    app = DoorbellApp()