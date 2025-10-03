Smart Face Registration and Recognition System

This project is a secure face registration and recognition system designed for access control and visitor verification. It combines modern AI-based models with a robust backend and user-friendly interfaces to provide a reliable, transparent, and secure solution.

Key Features
- Face Detection: YOLOv8 for real-time face localization.
- Face Embedding: face_recognition generates 128-D embeddings stored securely.
- Liveness Detection: MobileNetV3-Large model ensures the user is real and not a spoof (e.g., photo or video attack).
- Deepfake Detection: Fine-tuned Xception-41 model checks for AI-manipulated content.
- Yaw Pose Validation: MediaPipe Face Mesh verifies left, right, and center head turns for robust registration.
- Encryption & Security: All proof photos, embeddings, and videos are encrypted with AES before storage.
- Backend (FastAPI): Handles user registration, validation, logging, email notifications, and admin approval workflows.
- Frontend (React + Material UI): Provides an admin dashboard to manage registrations, view scores, and control user access.
- Recognition GUI (Tkinter): Real-time doorbell interface for verifying visitors against the database.

Workflow
- Visitor registers via webcam with three head poses (Center, Left, Right).
- Videos are processed for liveness and deepfake checks.
- Proof photo and embeddings are securely stored in an encrypted cache.
- Admin reviews results and can approve or block users from the dashboard.
- During recognition, live video feed is compared with stored embeddings for access control.

Dataset Used
For Liveness Dataset
<img width="907" height="797" alt="image" src="https://github.com/user-attachments/assets/c6f5bcec-4ca7-44d3-81c3-ee899bbbe3e0" />

For Deepfake Dataset
<img width="923" height="387" alt="image" src="https://github.com/user-attachments/assets/416e81fb-11c0-45b8-9775-766eb1305129" />
