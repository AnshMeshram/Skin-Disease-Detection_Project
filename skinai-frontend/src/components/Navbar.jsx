import React, { useState, useEffect } from "react";
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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const textColor = "#111827";
  const inactiveColor = "#111827";
  const tabBgContainer = "rgba(0,0,0,0.05)";

  return (
    <>
      <nav
        style={{
          position: "absolute",
          top: 12,
          left: "1rem",
          right: "1rem",
          zIndex: 100,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.75rem 1.25rem",
          background: "rgba(255, 255, 255, 0.75)",
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
            gap: "10px",
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
              flexShrink: 0,
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
              fontSize: "1.35rem",
              fontWeight: 800,
              color: textColor,
              cursor: "pointer",
              letterSpacing: "0.02em",
              whiteSpace: "nowrap",
            }}
          >
            Twacha<span style={{ color: "#10B981" }}>Rakshak</span>
          </div>
        </Link>

        {/* Desktop Tab Row */}
        <div
          className="desktop-nav-tabs"
          style={{
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
                padding: "8px 20px",
                fontSize: "0.85rem",
                fontWeight: 600,
                opacity: isActive ? 1 : 0.7,
                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
              })}
            >
              {tab.name}
            </NavLink>
          ))}
        </div>

        {/* Desktop API Status */}
        <div
          className="desktop-api-status"
          style={{
            alignItems: "center",
            gap: 8,
            background: "rgba(255, 255, 255, 0.8)",
            padding: "6px 12px",
            borderRadius: "999px",
            border: "1px solid rgba(0,0,0,0.05)",
            fontSize: "0.75rem",
            color: textColor,
            fontWeight: 700,
            letterSpacing: "0.02em",
            textTransform: "uppercase",
            whiteSpace: "nowrap",
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

        {/* Mobile Hamburger Button */}
        <button
          className="mobile-menu-btn"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle navigation menu"
          style={{
            background: "rgba(0, 0, 0, 0.05)",
            border: "none",
            borderRadius: "12px",
            padding: "8px",
            display: "none",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            color: textColor,
          }}
        >
          {mobileMenuOpen ? (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          ) : (
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="4" y1="6" x2="20" y2="6"></line>
              <line x1="4" y1="12" x2="20" y2="12"></line>
              <line x1="4" y1="18" x2="20" y2="18"></line>
            </svg>
          )}
        </button>
      </nav>

      {/* Mobile Drawer Menu */}
      {mobileMenuOpen && (
        <div
          style={{
            position: "fixed",
            top: 70,
            left: "1rem",
            right: "1rem",
            zIndex: 99,
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(24px)",
            borderRadius: "20px",
            padding: "1.25rem",
            boxShadow: "0 20px 40px rgba(0, 0, 0, 0.15)",
            border: "1px solid rgba(255, 255, 255, 0.6)",
            display: "flex",
            flexDirection: "column",
            gap: "10px",
            animation: "fadeUp 0.3s ease",
          }}
        >
          {TABS.map((tab) => (
            <NavLink
              key={tab.name}
              to={tab.path}
              onClick={() => setMobileMenuOpen(false)}
              style={({ isActive }) => ({
                background: isActive ? "#2563EB" : "rgba(0, 0, 0, 0.03)",
                color: isActive ? "#fff" : textColor,
                textDecoration: "none",
                borderRadius: "14px",
                padding: "12px 16px",
                fontSize: "0.95rem",
                fontWeight: 600,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              })}
            >
              <span>{tab.name}</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </NavLink>
          ))}

          {/* Mobile API status */}
          <div
            style={{
              marginTop: "8px",
              paddingTop: "12px",
              borderTop: "1px solid rgba(0,0,0,0.06)",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "0.8rem",
              fontWeight: 700,
              color: textColor,
              textTransform: "uppercase",
            }}
          >
            <span>API Status</span>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                background: isOnline ? "rgba(16, 185, 129, 0.1)" : "rgba(244, 63, 94, 0.1)",
                color: isOnline ? "#059669" : "#E11D48",
                padding: "4px 10px",
                borderRadius: "999px",
              }}
            >
              <div
                style={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  background: isOnline ? "#10B981" : "#F43F5E",
                }}
              />
              {isOnline ? "System Live" : "System Offline"}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
