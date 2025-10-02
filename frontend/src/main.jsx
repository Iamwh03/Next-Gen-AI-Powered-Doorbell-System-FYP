import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import App from "./InteractiveUI";


/* ─── One-time-token bootstrap ────────────────────────── */
const urlParams   = new URLSearchParams(window.location.search);
const incomingTok = urlParams.get("token");

if (incomingTok) {
  localStorage.setItem("regToken", incomingTok);
  // Strip the token from the address bar but keep it in storage
  window.history.replaceState({}, "", "/");
}
/* ─────────────────────────────────────────────────────── */

/* In dev you can skip <React.StrictMode> to avoid double renders */
ReactDOM.createRoot(document.getElementById("root")).render(
  // <React.StrictMode>
    <App />
  // </React.StrictMode>
);
