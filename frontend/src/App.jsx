import { BrowserRouter, Routes, Route } from "react-router-dom";
import NameEmailPage from "./NameEmailPage";
import RegisterPage  from "./RegisterPage";
const params = new URLSearchParams(window.location.search);
const TOKEN  = params.get("token");
if (TOKEN) localStorage.setItem("regToken", TOKEN);


export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"         element={<NameEmailPage />} />
        <Route path="/register" element={<RegisterPage  />} />
      </Routes>
    </BrowserRouter>
  );
}
