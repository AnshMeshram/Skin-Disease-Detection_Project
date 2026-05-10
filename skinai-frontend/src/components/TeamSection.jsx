import React from "react";
import { useReveal } from "../hooks/useReveal";
import { User, ShieldCheck } from "lucide-react";

const TEAM = [
  {
    name: "Prof. Uttam Chaskar",
    role: "Project Guide",
    isGuide: true,
    color: "#059669",
    image: "/images/uc.jpg",
  },
  {
    name: "Om Nagargoje",
    role: "ID: 612209031",
    color: "#2563EB",
    image: "/images/on.png",
  },
  {
    name: "Ansh Meshram",
    role: "ID: 612209029",
    color: "#2563EB",
    image: "/images/AnshMeshram_ProfileImage.jpg",
  },
  {
    name: "Ajinkya More",
    role: "ID: 612209030",
    color: "#2563EB",
    image: "/images/ajm.png",
  },
];

export default function TeamSection() {
  const ref = useReveal();
  return (
    <section
      id="students"
      className="reveal mesh-bg"
      ref={ref}
      style={{
        padding: "6rem 2rem 8rem",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          maxWidth: 1000,
          margin: "0 auto",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "4rem" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "#fff",
              border: "1px solid #E2E8F0",
              borderRadius: "999px",
              padding: "6px 16px",
              fontSize: "0.75rem",
              fontWeight: 800,
              color: "#64748B",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              boxShadow: "0 2px 4px rgba(0,0,0,0.02)",
            }}
          >
            <ShieldCheck size={14} color="#059669" /> Research & Development
            Team
          </div>
          <h2
            style={{
              fontFamily: "'Plus Jakarta Sans', sans-serif",
              fontSize: "2.5rem",
              fontWeight: 800,
              color: "#0F172A",
              marginTop: "1.25rem",
              letterSpacing: "-0.03em",
            }}
          >
            Project Contributors
          </h2>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "1.5rem",
          }}
        >
          {TEAM.map((m, i) => (
            <div
              key={i}
              style={{
                background: "#fff",
                border: "1px solid #F1F5F9",
                borderRadius: "24px",
                padding: "2.5rem 1.5rem",
                textAlign: "center",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                transition: "all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
                boxShadow: "0 10px 30px -10px rgba(0,0,0,0.04)",
                position: "relative",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-8px)";
                e.currentTarget.style.boxShadow =
                  "0 20px 40px -15px rgba(0,0,0,0.1)";
                e.currentTarget.style.borderColor = m.color + "30";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow =
                  "0 10px 30px -10px rgba(0,0,0,0.04)";
                e.currentTarget.style.borderColor = "#F1F5F9";
              }}
            >
              <div
                style={{
                  width: 80,
                  height: 80,
                  borderRadius: "50%",
                  background: `${m.color}08`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: "1.5rem",
                  border: `2px solid ${m.color}20`,
                  overflow: "hidden",
                }}
              >
                {m.image ? (
                  <img
                    src={m.image}
                    alt={m.name}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      objectPosition: "center 20%",
                    }}
                  />
                ) : (
                  <User size={32} color={m.color} strokeWidth={1.5} />
                )}
              </div>
              <div
                style={{
                  fontFamily: "Outfit, sans-serif",
                  fontSize: "1.15rem",
                  fontWeight: 700,
                  color: "#0F172A",
                  marginBottom: "4px",
                }}
              >
                {m.name}
              </div>
              <div
                style={{
                  fontSize: "0.7rem",
                  color: m.color,
                  fontWeight: 800,
                  textTransform: "uppercase",
                  letterSpacing: "0.1em",
                }}
              >
                {m.role}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
