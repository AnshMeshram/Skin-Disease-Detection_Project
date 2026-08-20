import React from "react";
import { useReveal } from "../hooks/useReveal";
import { User, ShieldCheck, Award, GraduationCap, Sparkles } from "lucide-react";
import FloatingParticlesBackground from "./FloatingParticlesBackground";

const TEAM = [
  {
    name: "Prof. Uttam Chaskar",
    role: "Project Guide & Faculty Mentor",
    id: "Faculty Mentor",
    specialty: "Instrumentation & Control",
    isGuide: true,
    color: "#059669",
    image: "/images/uc.jpg",
  },
  {
    name: "Om Nagargoje",
    role: "Student Researcher",
    id: "ID: 612209031",
    specialty: "Deep Learning & 3D-CA Attention",
    color: "#2563EB",
    image: "/images/on.png",
  },
  {
    name: "Ansh Meshram",
    role: "Student Researcher",
    id: "ID: 612209029",
    specialty: "Full-Stack AI & Preprocessing",
    color: "#2563EB",
    image: "/images/AnshMeshram_ProfileImage.jpg",
  },
  {
    name: "Ajinkya More",
    role: "Student Researcher",
    id: "ID: 612209030",
    specialty: "ISIC Benchmark & Evaluation",
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
        padding: "6rem 2rem",
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Dynamic Neural Particles Background behind Contributors */}
      <FloatingParticlesBackground 
        variant="neural" 
        count={18} 
        colors={['#2563EB', '#10B981', '#6366F1', '#38BDF8']} 
        opacity={0.25} 
        speed={0.3} 
      />

      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          position: "relative",
          zIndex: 1,
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "3.5rem" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "#fff",
              border: "1px solid #E5E7EB",
              borderRadius: "999px",
              padding: "6px 16px",
              fontSize: "0.75rem",
              fontWeight: 800,
              color: "#64748B",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
            }}
          >
            <ShieldCheck size={14} color="#059669" /> Research & Development Team
          </div>
          <h2
            className="syne"
            style={{
              fontSize: "1.75rem",
              fontWeight: 800,
              color: "#111827",
              marginTop: "1.25rem",
              letterSpacing: "-0.01em",
            }}
          >
            Project Contributors
          </h2>
          <p style={{ fontSize: "0.9375rem", color: "#6B7280", maxWidth: 620, margin: "0.5rem auto 0" }}>
            Developed at COEP Technological University with advanced ensemble deep learning and clinical decision-support architectures.
          </p>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
            gap: "1.5rem",
          }}
        >
          {TEAM.map((m, i) => (
            <div
              key={i}
              style={{
                background: "rgba(255, 255, 255, 0.92)",
                backdropFilter: "blur(16px)",
                border: "1px solid #E5E7EB",
                borderRadius: "24px",
                padding: "2.25rem 1.5rem",
                textAlign: "center",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                transition: "all 0.35s cubic-bezier(0.175, 0.885, 0.32, 1.275)",
                boxShadow: "0 4px 20px rgba(0,0,0,0.03)",
                position: "relative",
                overflow: "hidden",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = "translateY(-6px)";
                e.currentTarget.style.boxShadow = `0 16px 36px -8px ${m.color}25`;
                e.currentTarget.style.borderColor = m.color + "50";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = "translateY(0)";
                e.currentTarget.style.boxShadow = "0 4px 20px rgba(0,0,0,0.03)";
                e.currentTarget.style.borderColor = "#E5E7EB";
              }}
            >
              {/* Top Guide / Lead Badge */}
              {m.isGuide && (
                <div style={{
                  position: "absolute",
                  top: 12,
                  right: 12,
                  background: "#ECFDF5",
                  color: "#059669",
                  border: "1px solid #A7F3D0",
                  borderRadius: "999px",
                  padding: "2px 8px",
                  fontSize: "0.6875rem",
                  fontWeight: 800,
                  display: "flex",
                  alignItems: "center",
                  gap: "4px",
                }}>
                  <Sparkles size={10} /> Faculty Guide
                </div>
              )}

              {/* Avatar with dynamic aura ring */}
              <div
                style={{
                  width: 88,
                  height: 88,
                  borderRadius: "50%",
                  background: `${m.color}10`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: "1.25rem",
                  border: `2.5px solid ${m.isGuide ? '#059669' : '#2563EB'}`,
                  boxShadow: `0 0 0 4px ${m.isGuide ? 'rgba(5,150,105,0.1)' : 'rgba(37,99,235,0.1)'}`,
                  overflow: "hidden",
                  position: "relative",
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
                  <User size={36} color={m.color} strokeWidth={1.5} />
                )}
              </div>

              {/* Name */}
              <div
                className="syne"
                style={{
                  fontSize: "1.0625rem",
                  fontWeight: 700,
                  color: "#111827",
                  marginBottom: "4px",
                }}
              >
                {m.name}
              </div>

              {/* Student ID / Guide Tag */}
              <div
                style={{
                  fontSize: "0.72rem",
                  color: m.color,
                  fontWeight: 800,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  marginBottom: "0.75rem",
                }}
              >
                {m.id}
              </div>

              {/* Specialty Pill */}
              <div style={{
                background: "#F8FAFC",
                border: "1px solid #E2E8F0",
                borderRadius: "999px",
                padding: "4px 12px",
                fontSize: "0.72rem",
                fontWeight: 600,
                color: "#475569",
                marginTop: "auto",
              }}>
                {m.specialty}
              </div>
            </div>
          ))}
        </div>

        {/* Institutional Footnote Banner */}
        <div style={{
          marginTop: "3rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "10px",
          color: "#6B7280",
          fontSize: "0.8125rem",
          fontWeight: 600,
        }}>
          <GraduationCap size={18} color="#2563EB" />
          <span>COEP Technological University, Pune • Department of Instrumentation &amp; Control</span>
        </div>
      </div>
    </section>
  );
}
