import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";

/* ------------------------------------------------------------------ */
/* Config – change only these two lines if you move the backend URL   */
const API   = import.meta.env.VITE_API_URL || "http://localhost:8003";
const poses = ["Center", "Left", "Right"];
const RECORD_MS = 6000;
/* ------------------------------------------------------------------ */

export default function RegisterPage() {
  const { state } = useLocation();      // { name, email } from the first page
  const videoRef  = useRef(null);

  const [step,  setStep]  = useState(0);     // 0,1,2   (3 = done, ready to upload)
  const [busy,  setBusy]  = useState(false);
  const [msg,   setMsg]   = useState("");
  const [blobs, setBlobs] = useState({});    // { Center: Blob, Left: Blob, Right: Blob }

  /* ──────────────────────── 1. Grab webcam once ─────────────────────── */
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then(stream => (videoRef.current.srcObject = stream))
      .catch(() => alert("Could not access webcam"));
  }, []);

  /* ──────────────────────── 2. Record & yaw-check ────────────────────── */
  const recordPose = () => {
    setBusy(true);

    const recorder = new MediaRecorder(videoRef.current.srcObject, {
      mimeType: "video/webm",
    });
    const chunks = [];

    recorder.ondataavailable = e => chunks.push(e.data);

    recorder.onstop = async () => {
      const blob = new Blob(chunks, { type: "video/webm" });

      /* ---- 2-A. Ask backend if this clip passes yaw ---- */
      const fd = new FormData();
      fd.append("video", blob, `${poses[step]}.webm`);

      const yawRes = await fetch(`${API}/validate_yaw/${poses[step]}`, {
        method: "POST",
        body: fd,
      }).then(r => r.json());

      if (!yawRes.pass) {
        alert("Head angle not correct – please try that pose again!");
        setBusy(false);          // stay on the same step
        return;
      }

      /* ---- 2-B. Yaw OK → store blob & advance ---- */
      setBlobs(prev => ({ ...prev, [poses[step]]: blob }));
      setStep(s => s + 1);
      setBusy(false);
    };

    recorder.start();
    setTimeout(() => recorder.stop(), RECORD_MS);   // stop after 3 seconds
  };

  /* ──────────────────────── 3. Upload all three clips ────────────────── */
  const uploadAll = async () => {
    setBusy(true);
    const fd = new FormData();
    fd.append("name",  state.name);
    fd.append("email", state.email);
    fd.append("center", blobs.Center, "Center.webm");
    fd.append("left",   blobs.Left,   "Left.webm");
    fd.append("right",  blobs.Right,  "Right.webm");

    const { uid } = await fetch(`${API}/register`, { method: "POST", body: fd })
                          .then(r => r.json());

    /* ---- poll /status every 3 s until backend finishes ---- */
    const iv = setInterval(async () => {
      const status = await fetch(`${API}/status/${uid}`).then(r => r.json());
      if (status.status === "done") {
        clearInterval(iv);
        setBusy(false);
        setMsg("✅ Finished! Check your e-mail.");
      }
    }, 3000);
  };

  /* ──────────────────────────── 4. UI ──────────────────────────────── */
  if (msg) {
    return <div style={{ padding: 32, fontSize: 24 }}>{msg}</div>;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
        alignItems: "center",
        padding: 32,
      }}
    >
      <h2>Pose: {poses[step] ?? "All done!"}</h2>

      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{
            border: "1px solid #ccc",
            transform: "scaleX(-1)"   // mirror only the preview
        }}
      />

      {!busy && step < 3 && (
        <button onClick={recordPose} style={{ padding: "8px 24px" }}>
          Record {RECORD_MS / 1000} s
        </button>
      )}

      {!busy && step === 3 && (
        <button
          onClick={uploadAll}
          style={{
            padding: "8px 24px",
            background: "#2563eb",
            color: "#fff",
            border: "none",
            borderRadius: 6,
          }}
        >
          Upload &amp; Finish
        </button>
      )}

      {busy && <p>⏳ Working…</p>}
    </div>
  );
}
