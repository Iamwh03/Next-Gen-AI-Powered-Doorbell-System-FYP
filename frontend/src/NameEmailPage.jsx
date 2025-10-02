import { useState } from "react";
import { useNavigate } from "react-router-dom";

export default function NameEmailPage() {
  const nav = useNavigate();
  const [name,  setName]  = useState("");
  const [email, setEmail] = useState("");

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16,alignItems:"center",padding:32}}>
      <h1>Start Face Registration</h1>

      <input
        style={{padding:8,width:240}}
        placeholder="Name"
        value={name}
        onChange={e => setName(e.target.value)}
      />

      <input
        style={{padding:8,width:240}}
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
      />

      <button
        style={{padding:"8px 24px",background:"#2563eb",color:"#fff",border:"none",borderRadius:6}}
        onClick={() => nav("/register", { state: { name, email } })}
        disabled={!name || !email}
      >
        Next
      </button>
    </div>
  );
}
