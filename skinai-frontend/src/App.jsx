import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import './index.css';

import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Model from './pages/Model';
import SkinGuide from './pages/SkinGuide';
import Research from './pages/Research';
import { checkHealth } from './api';

// Scroll to top on route change
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("offline");

  /* ── Poll health every 10 s ─────────────────── */
  useEffect(() => {
    const poll = async () => {
      try {
        await checkHealth();
        setApiStatus("online");
      } catch {
        setApiStatus("offline");
      }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => clearInterval(id);
  }, []);

  return (
    <BrowserRouter>
      <ScrollToTop />
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar apiStatus={apiStatus} />
        
        <div style={{ flex: 1 }}>
          <Routes>
            <Route path="/" element={<Home apiOnline={apiStatus === "online"} />} />
            <Route path="/model" element={<Model />} />
            <Route path="/guide" element={<SkinGuide />} />
            <Route path="/research" element={<Research />} />
          </Routes>
        </div>

        <Footer />
      </div>
    </BrowserRouter>
  );
}
