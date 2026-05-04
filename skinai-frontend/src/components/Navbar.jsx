import React from "react";
import { NavLink, Link, useLocation } from "react-router-dom";

const TABS = [
  { name: "Home", path: "/" },
  { name: "Skin Guide", path: "/guide" },
  { name: "Model Info", path: "/model" },
  { name: "Research", path: "/research" },
];

export default function Navbar({ apiStatus }) {
  const location = useLocation();
  const isHome = location.pathname === "/";
  const isOnline = apiStatus === "online";

  // Updated for bright theme compatibility
  const textColor = "#111827";
  const inactiveColor = "#111827";
  const tabBgContainer = "rgba(0,0,0,0.05)";

  return (
    <nav
      style={{
        position: "absolute",
        top: 12,
        left: "2rem",
        right: "2rem",
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0.75rem 2rem",
        background: "rgba(255, 255, 255, 0.7)",
        backdropFilter: "blur(20px)",
        borderRadius: "24px",
        border: "1px solid rgba(255, 255, 255, 0.5)",
        boxShadow: "0 8px 32px rgba(0, 0, 0, 0.05)",
      }}
    >
      {/* Brand */}
      <Link
        to="/"
        style={{
          textDecoration: "none",
          display: "flex",
          alignItems: "center",
          gap: "12px",
        }}
      >
        <div
          style={{
            width: "36px",
            height: "36px",
            borderRadius: "10px",
            background: "linear-gradient(135deg, #10B981 0%, #059669 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 4px 15px rgba(16, 185, 129, 0.4)",
          }}
        >
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#fff"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
          </svg>
        </div>
        <div
          style={{
            fontFamily: "Outfit, sans-serif",
            fontSize: "1.5rem",
            fontWeight: 800,
            color: textColor,
            cursor: "pointer",
            letterSpacing: "0.02em",
            transition: "color 0.3s ease",
          }}
        >
          Twacha<span style={{ color: "#10B981" }}>Rakshak</span>
        </div>
      </Link>

      {/* Tab row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "6px",
          background: tabBgContainer,
          backdropFilter: "blur(12px)",
          borderRadius: "999px",
          padding: "6px",
          transition: "background 0.3s ease",
        }}
      >
        {TABS.map((tab) => (
          <NavLink
            key={tab.name}
            to={tab.path}
            className={isHome ? "nav-hover-home" : "nav-hover-other"}
            style={({ isActive }) => ({
              background: isActive ? "#2563EB" : "transparent",
              color: isActive ? "#fff" : inactiveColor,
              textDecoration: "none",
              borderRadius: "999px",
              padding: "8px 24px",
              fontSize: "0.85rem",
              fontWeight: 600,
              opacity: isActive ? 1 : 0.6,
              transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            })}
          >
            {tab.name}
          </NavLink>
        ))}
      </div>

      {/* API status */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          background: "rgba(255, 255, 255, 0.8)",
          padding: "6px 14px",
          borderRadius: "999px",
          border: "1px solid rgba(0,0,0,0.05)",
          fontSize: "0.75rem",
          color: textColor,
          fontWeight: 700,
          letterSpacing: "0.02em",
          textTransform: "uppercase",
        }}
      >
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: isOnline ? "#10B981" : "#F43F5E",
            boxShadow: isOnline
              ? "0 0 12px rgba(16, 185, 129, 0.6)"
              : "0 0 12px rgba(244, 63, 94, 0.6)",
            animation: "pulse 2s infinite",
          }}
        />
        {isOnline ? "System Live" : "System Offline"}
      </div>
    </nav>
  );
}
