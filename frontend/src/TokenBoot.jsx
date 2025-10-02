/* 📄 TokenBoot.jsx  (or just put in the same file) */
import { useEffect } from "react";

export default function TokenBoot() {
  useEffect(() => {
    const urlTok = new URLSearchParams(window.location.search).get("token");
    if (urlTok) localStorage.setItem("regToken", urlTok);
  }, []);            // ← run exactly once on first render
  return null;       // nothing visible
}

