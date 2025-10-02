import TokenBoot from "./TokenBoot";
import axios from "axios";
import { useState, useRef, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import styled, { createGlobalStyle, keyframes, css  } from "styled-components";
import { Loader2, CheckCircle2, User, Mail, XCircle } from "lucide-react";

let recording = false;
const API = import.meta.env.VITE_API_URL || "http://localhost:8003";
const poses = ["Center", "Left", "Right"];
const RECORD_MS = 3000;
const getToken = () => localStorage.getItem("regToken");
const addTok = (path) => {
  const tok = getToken();
  return `${path}${path.includes("?") ? "&" : "?"}token=${tok ?? ""}`;
};

const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: #F7F8FA;
    color: #1F2937;
  }
`;

const PageContainer = styled.div`
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
`;
const scanFlow = keyframes`
  0% { transform: translateX(0); opacity: 0.8; }
  50% { transform: translateX(100%); opacity: 1; }
  100% { transform: translateX(0); opacity: 0.8; }
`;
const sweepDown = keyframes`
  0% { transform: translateY(-100%); }
  100% { transform: translateY(200%); }
`;

const sweepLeftToRight = keyframes`
  0% { transform: translateX(-100%); }
  100% { transform: translateX(200%); }
`;

const sweepRightToLeft = keyframes`
  0% { transform: translateX(200%); }
  100% { transform: translateX(-100%); }
`;

const VideoContainer = styled.div`
  position: relative;
  width: 100%;
  max-width: 450px;           /* restrict width to make it more phone-like */
  margin: 0 auto;
  border-radius: 1.5rem;
  overflow: hidden;
  border: 1px solid #E5E7EB;
  box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
  aspect-ratio: 9 / 12;        /* portrait mode */
`;

const OverlayCanvas = styled.div`
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: linear-gradient(180deg, rgba(0,0,0,0.3) 0%, rgba(0,0,0,0.1) 40%, rgba(0,0,0,0.1) 60%, rgba(0,0,0,0.3) 100%);
  pointer-events: none;
`;

const FaceOval = styled.div`
  width: 300px;  /* Increased size */
  height: 400px; /* Increased size */
  border: 2px solid rgba(255, 255, 255, 0.7);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(59, 130, 246, 0.5); /* Glowing effect */
  position: relative;
`;

const verticalSweep = keyframes`
  0% { transform: translateY(-100%); }
  100% { transform: translateY(100%); }
`;

const getScanAnimation = (pose) => {
  switch (pose) {
    case "Left":
      return css`
        width: 50%;
        height: 100%;
        background: linear-gradient(to left, rgba(59, 130, 246, 0) 0%, rgba(59, 130, 246, 0.6) 50%, rgba(59, 130, 246, 0) 100%);
        animation: ${sweepRightToLeft} ${RECORD_MS / 1000}s ease-in-out forwards;
      `;
    case "Right":
      return css`
        width: 50%;
        height: 100%;
        background: linear-gradient(to right, rgba(59, 130, 246, 0) 0%, rgba(59, 130, 246, 0.6) 50%, rgba(59, 130, 246, 0) 100%);
        animation: ${sweepLeftToRight} ${RECORD_MS / 1000}s ease-in-out forwards;
      `;
    default: // "Center"
      return css`
        width: 100%;
        height: 50%;
        background: linear-gradient(to bottom, rgba(59, 130, 246, 0) 0%, rgba(59, 130, 246, 0.6) 50%, rgba(59, 130, 246, 0) 100%);
        animation: ${sweepDown} ${RECORD_MS / 1000}s ease-in-out forwards;
      `;
  }
};

const ScanLine = styled.div`
  position: absolute;
  top: 0;
  left: 0;
  ${({ pose }) => getScanAnimation(pose)}
`;


const InstructionContainer = styled(motion.div)`
  /* REMOVED absolute positioning */
  text-align: center;
  margin-bottom: 1.5rem; /* Add space between text and video */
`;

const InstructionText = styled(motion.h2)`
  font-size: 1.5rem; /* 24px */
  font-weight: 600;
  color: #1F2937; /* Change to dark color */
  margin: 0;
`;

const InstructionSubtext = styled(motion.p)`
  font-size: 1rem; /* 16px */
  color: #6B7280; /* Change to grey color */
  margin: 0.25rem 0 0 0;
  /* REMOVED text-shadow and opacity */
`;



const Card = styled.div`
  width: 100%;
  max-width: 520px;      /* ← wider card (was 448px)           */
  background: #ffffff;
  border: 1px solid #E5E7EB;
  border-radius: 1.5rem; /* ← 24 px → 1.5 rem keeps it tidy    */
  padding: 3rem 2.5rem;  /* ← more vertical breathing room     */
  box-shadow:
    0 25px 40px -10px rgba(0,0,0,0.04),
    0  8px 18px -6px  rgba(0,0,0,0.06);
`;

const MotionButton = styled(motion.button)`
  width: 100%;
  padding: 0.9rem 0;
  border-radius: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;

  background: ${p =>
    p.disabled
      ? "#E5E7EB"
      : "linear-gradient(90deg,#4F46E5 0%,#6366F1 100%)"}; /* indigo */

  color: #ffffff;
  border: none;

  &:not(:disabled):hover { transform: translateY(-2px); }
`;


const Title = styled(motion.h1)`
  font-size: 2rem;  /* 32 px */
  font-weight: 800; /* extra-bold */
  text-align: center;
  margin-bottom: 0.65rem;  /* instead of margin-top 0 */
`;


const Subtitle = styled(motion.p)`
  color: #6B7280;
  text-align: center;
  margin-bottom: 2rem;
`;

const ErrorText = styled.div`
  color: #EF4444;
  margin-top: 0.5rem;
  font-size: 0.875rem;
`;

const InputWrapper = styled.div` position: relative; `;
const IconWrapper = styled.div`
  position: absolute;
  top: 50%;
  left: 0.75rem;        /* tighter */
  transform: translateY(-50%);
  color: #9CA3AF;
  pointer-events: none;

  svg { width: 18px; height: 18px; } /* smaller icon */
`;

const StyledInput = styled.input`
  width: 83%;
  padding: 0.9rem 1rem 0.9rem 3rem;
  background: #ffffff;              /* was #F9FAFB */
  border: 1px solid #D1D5DB;
  border-radius: 0.75rem;
  font-size: 1rem;

  &:disabled {
    background: #F3F4F6;            /* a softer grey */
    color: #6B7280;
  }
`;


function IconInput({ value, onChange, placeholder, icon, disabled }) {
  return (
    <InputWrapper>
      {icon && <IconWrapper>{icon}</IconWrapper>}
      <StyledInput
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        disabled={disabled}
      />
    </InputWrapper>
  );
}

const pageAnim = {
  hidden: { opacity: 0, y: 20 },
  enter: { opacity: 1, y: 0, transition: { duration: 0.4 } },
  exit:  { opacity: 0, y: -20, transition: { duration: 0.3 } },
};
const containerAnim = { hidden: {}, show: { transition: { staggerChildren: 0.1 } } };
const itemAnim = {
  hidden: { y: 20, opacity: 0 },
  show:   { y: 0, opacity: 1, transition: { type: "spring", stiffness: 100 } },
};

export default function App() {
  return (
    <Router>
      <TokenBoot />
      <GlobalStyle />
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<NameEmailPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/thank-you" element={<ThankYouPage />} />
        </Routes>
      </AnimatePresence>
    </Router>
  );
}

function NameEmailPage() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [mail, setMail] = useState("");
  const [error, setError] = useState("");

  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || localStorage.getItem("regToken");

  useEffect(() => {
    if (!token) return setError("Missing invitation token.");
    axios
      .get(`${API}/validate_token?token=${token}`)
      .then(({ data }) => setMail(data.email))
      .catch(() => setError("Invalid or expired invitation link."));
  }, [token]);

  const ready = name && mail;

  return (
    <PageContainer>
      <motion.div variants={pageAnim} initial="hidden" animate="enter" exit="exit">
        <Card $maxWidth="768px">
          <motion.div variants={containerAnim} initial="hidden" animate="show">
            <Title variants={itemAnim}>Create Your Account</Title>
            <Subtitle variants={itemAnim}>Begin your secure registration.</Subtitle>

            <motion.div variants={itemAnim} style={{ marginBottom: "1.5rem" }}>
              <IconInput
                placeholder="Name"
                value={name}
                onChange={e => setName(e.target.value)}
                icon={<User size={20} />}
              />
            </motion.div>

            <motion.div variants={itemAnim} style={{ marginBottom: "2.5rem" }}>
              <IconInput
                placeholder="name@example.com"
                value={mail}
                disabled
                icon={<Mail size={20} />}
              />
              {error && <ErrorText>{error}</ErrorText>}
            </motion.div>

            <motion.div variants={itemAnim}>
              <MotionButton
                disabled={!ready}
                whileTap={ready ? { scale: 0.98 } : {}}
                onClick={() =>
                  nav(`/register?token=${token}`, {
                    state: { name, email: mail },
                  })
                }
              >
                Secure Registration
              </MotionButton>
            </motion.div>
          </motion.div>
        </Card>
      </motion.div>
    </PageContainer>
  );
}

function RegisterPage() {
  const nav = useNavigate();
  const storedTok =
    localStorage.getItem("regToken") ||
    new URLSearchParams(window.location.search).get("token");
  if (!storedTok) {
    return (
      <PageContainer>
        <Card>
          <p style={{ padding: 40, textAlign: "center", fontSize: 18 }}>
            Invalid or already-used link.
          </p>
        </Card>
      </PageContainer>
    );
  }

  const { state } = useLocation();
  const videoRef = useRef(null);
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [blobs, setBlobs] = useState({});
  const [showError, setShowError] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  // You can define instructions here for easy management
  const instructions = [
    { title: "Look Straight Ahead", subtitle: "Position your face inside the oval." },
    { title: "Turn Your Head Left", subtitle: "Slowly turn to your left." },
    { title: "Turn Your Head Right", subtitle: "Now, slowly turn to your right." },
    { title: "All Set!", subtitle: "Click below to complete your registration." }
  ];
  const displayError = (msg) => {
    setErrMsg(msg);
    setShowError(true);
  };

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1920, height: 1280 } })
      .then((stream) => (videoRef.current.srcObject = stream))
      .catch(() =>
        displayError("Could not access the camera. Please check your permissions.")
      );
  }, []);

  const recordPose = () => {
    if (recording) return;
    recording = true;
    setBusy(true);

    const rec = new MediaRecorder(videoRef.current.srcObject, {
      mimeType: "video/webm",
      videoBitsPerSecond: 2_500_000
    });
    const chunks = [];

    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      const blob = new Blob(chunks, { type: "video/webm" });
      const fd = new FormData();
      fd.append("video", blob, `${poses[step]}.webm`);

      try {
        const res = await fetch(
          addTok(`${API}/validate_yaw/${poses[step]}`),
          { method: "POST", body: fd }
        );
        res._json = res._json || res.json();
        const { pass } = await res._json;

        if (pass) {
          setBlobs((o) => ({ ...o, [poses[step]]: blob }));
          setStep((s) => s + 1);
        } else {
          displayError("Pose not detected correctly. Please adjust and retry.");
        }
      } catch (err) {
        console.error(err);
        displayError(err.message || "Upload failed");
      } finally {
        setBusy(false);
        recording = false;
      }
    };

    rec.start();
    setTimeout(() => rec.stop(), RECORD_MS + 100);
  };

  const uploadAll = async () => {
    nav("/thank-you");
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("name", state.name);
      fd.append("email", state.email);
      poses.forEach((p) => fd.append(p.toLowerCase(), blobs[p], `${p}.webm`));

      const res = await fetch(addTok(`${API}/register`), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());

      // immediately navigate:


      // still poll in background if you like:
      const { uid } = await res.json();
      const poll = setInterval(async () => {
        const s = await fetch(addTok(`${API}/status/${uid}`)).then((r) =>
          r.json()
        );
        if (s.status === "done") {
          clearInterval(poll);
          setBusy(false);
          localStorage.removeItem("regToken");
        }
      }, 3000);
    } catch (err) {
      console.error(err);
      displayError(err.message);
      setBusy(false);
    }
  };

  return (
    <>
      <PageContainer>
        <motion.div variants={pageAnim} initial="hidden" animate="enter" exit="exit">
          {/* Change the Card's maxWidth and move the instructions */}
          <Card $maxWidth="560px" style={{ padding: '2.5rem' }}>

            {/* INSTRUCTIONS ARE MOVED HERE (Outside the video) */}
            <AnimatePresence mode="wait">
              <InstructionContainer
                key={step}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
              >
                <InstructionText>{instructions[step].title}</InstructionText>
                <InstructionSubtext>{instructions[step].subtitle}</InstructionSubtext>
              </InstructionContainer>
            </AnimatePresence>

            <VideoContainer>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  transform: "scaleX(-1)",
                }}
              />
              <OverlayCanvas>
                <FaceOval>
                  {busy && step < poses.length && <ScanLine key={step} pose={poses[step]} />}
                </FaceOval>
                {/* The InstructionContainer is no longer here */}
              </OverlayCanvas>
            </VideoContainer>

            <div style={{ display: "flex", justifyContent: "center", marginTop: "2rem" }}>
              {busy ? (
                <Loader2 size={32} className="animate-spin" color="#3B82F6" />
              ) : step < poses.length ? (
                <MotionButton onClick={recordPose}>Record Pose</MotionButton>
              ) : (
                <MotionButton onClick={uploadAll}>Finish & Upload</MotionButton>
              )}
            </div>
          </Card>
        </motion.div>
      </PageContainer>
      <AnimatePresence>
        {showError && (
          <ErrorModal message={errMsg} onClose={() => setShowError(false)} />
        )}
      </AnimatePresence>
    </>
  );
}

const ModalBackdrop = styled(motion.div)`
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.4);
  display: flex; align-items: center; justify-content: center;
`;

const ModalContent = styled(motion.div)`
  background: white; border-radius: 12px; padding: 2rem; max-width: 400px;
`;

function ErrorModal({ message, onClose }) {
  return (
    <ModalBackdrop onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()}>
        <XCircle size={48} color="#EF4444" />
        <h2>An Error Occurred</h2>
        <p>{message}</p>
        <MotionButton
          onClick={onClose}
          style={{ background: "linear-gradient(to right,#EF4444,#F87171)" }}
        >
          Close
        </MotionButton>
      </ModalContent>
    </ModalBackdrop>
  );
}

function ThankYouPage() {
  return (
    <PageContainer>
      <Card $maxWidth="448px">
        <div style={{ textAlign: "center" }}>
          <CheckCircle2
            size={64}
            color="#34D399"
            style={{ margin: "0 auto 1rem" }}
          />
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, margin: "1rem 0" }}>
            Thank you!
          </h2>
          <p style={{ color: "#4B5563", lineHeight: 1.6 }}>
            Your registration has been submitted.<br />
            We’ll notify you once it’s processed.
          </p>
        </div>
      </Card>
    </PageContainer>
  );
}

