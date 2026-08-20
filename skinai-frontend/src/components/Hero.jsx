import { useRef, useState, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Camera,
  Upload,
  Search,
  RotateCcw,
  User,
  Users,
  Hash,
  X,
  ShieldCheck,
  AlertCircle,
} from "lucide-react";
import { validatePatientInfo } from "../schemas/patientSchema";
import FloatingParticlesBackground from "./FloatingParticlesBackground";

const MODEL_BADGES = [
  { name: "EfficientNet-B3", icon: "⚡" },
  { name: "Inception V3", icon: "🔬" },
  { name: "ConvNeXt Tiny", icon: "🧠" },
];

export default function Hero({
  onAnalyze,
  loading,
  patientInfo,
  setPatientInfo,
}) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [genderOpen, setGenderOpen] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [zoom, setZoom] = useState(1);

  // Validate patient metadata with Zod
  const validation = validatePatientInfo(patientInfo || {});

  const fileRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const loadFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    stopCamera();
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPatientInfo({ name: "", gender: "", age: "" });
    setZoom(1);
    stopCamera();
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      alert(
        "Camera access is not supported by your browser or connection (requires HTTPS or localhost).",
      );
      return;
    }

    try {
      // Try to get stream first to verify hardware exists
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setIsCameraActive(true);

      // Use a slightly longer delay to ensure the video element is mounted in the DOM
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play().catch(console.error);
        }
      }, 100);
    } catch (err) {
      console.warn("Camera access failed:", err);
      setIsCameraActive(false);
      alert("Could not open the camera. Check permissions or use Upload File.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const captureFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Calculate cropped dimensions for zoom
    const w = video.videoWidth;
    const h = video.videoHeight;
    const zoomW = w / zoom;
    const zoomH = h / zoom;
    const startX = (w - zoomW) / 2;
    const startY = (h - zoomH) / 2;

    canvas.width = 640; // Standardize output size
    canvas.height = 480;
    const ctx = canvas.getContext("2d");

    // Draw the zoomed portion of the video onto the canvas
    ctx.drawImage(
      video,
      startX,
      startY,
      zoomW,
      zoomH,
      0,
      0,
      canvas.width,
      canvas.height,
    );

    canvas.toBlob(
      (blob) => {
        if (blob) {
          const f = new File([blob], "capture.jpg", { type: "image/jpeg" });
          loadFile(f);
          stopCamera();
        }
      },
      "image/jpeg",
      0.95,
    );
  };

  useEffect(() => {
    const handleOutside = () => setGenderOpen(false);
    if (genderOpen) window.addEventListener("click", handleOutside);
    return () => {
      window.removeEventListener("click", handleOutside);
      stopCamera();
    };
  }, [genderOpen]);

  return (
    <section
      id="home"
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
      }}
    >
      {/* ── Background Photo ── */}
      <img
        src="/hero-bg.png"
        alt=""
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          objectFit: "cover",
          filter: "brightness(0.9) blur(1px)",
          zIndex: 0,
        }}
      />
      <div
        style={{
          position: "absolute",
          inset: 0,
          zIndex: 1,
          background:
            "linear-gradient(to bottom, rgba(0,0,0,0.05) 0%, rgba(0,0,0,0.2) 100%)",
        }}
      />

      {/* ── Content ── */}
      <div
        style={{
          position: "relative",
          zIndex: 2,
          maxWidth: 700,
          width: "100%",
          padding: "8rem 2rem 2rem",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          textAlign: "center",
          gap: "1.25rem",
        }}
      >
        {/* ── 3D Glassmorphism Title Card ── */}
        <div
          style={{
            position: "relative",
            maxWidth: 680,
            width: "100%",
          }}
        >
          {/* Ambient 3D Glow Orbs behind the card */}
          <div
            aria-hidden="true"
            style={{
              position: "absolute",
              top: "-15%",
              left: "-10%",
              width: "120%",
              height: "130%",
              background:
                "radial-gradient(circle at 20% 30%, rgba(37, 99, 235, 0.22) 0%, transparent 55%), radial-gradient(circle at 80% 70%, rgba(16, 185, 129, 0.18) 0%, transparent 55%), radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.12) 0%, transparent 60%)",
              filter: "blur(24px)",
              pointerEvents: "none",
              zIndex: 0,
            }}
          />

          {/* Main 3D Hyper-Realistic Frosted Glass Container */}
          <div
            style={{
              position: "relative",
              zIndex: 1,
              background:
                "linear-gradient(135deg, rgba(255, 255, 255, 0.75) 0%, rgba(255, 255, 255, 0.38) 45%, rgba(255, 255, 255, 0.62) 75%, rgba(255, 255, 255, 0.3) 100%)",
              backdropFilter: "blur(40px) saturate(200%)",
              WebkitBackdropFilter: "blur(40px) saturate(200%)",
              border: "1.5px solid rgba(255, 255, 255, 0.85)",
              borderRadius: "32px",
              padding: "2.5rem 2.5rem 2.25rem",
              boxShadow:
                "0 35px 70px -15px rgba(15, 23, 42, 0.12), 0 20px 40px -10px rgba(37, 99, 235, 0.14), inset 0 2px 3px rgba(255, 255, 255, 0.98), inset 0 -2px 4px rgba(37, 99, 235, 0.12), inset 2px 0 3px rgba(255, 255, 255, 0.8), inset 0 0 28px rgba(255, 255, 255, 0.45)",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "1.15rem",
              overflow: "hidden",
            }}
          >
            {/* 3D Diagonal Specular Glass Reflection Sheen */}
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "60%",
                background:
                  "linear-gradient(120deg, rgba(255, 255, 255, 0.55) 0%, rgba(255, 255, 255, 0.22) 35%, transparent 65%)",
                borderRadius: "32px 32px 140px 32px",
                pointerEvents: "none",
                zIndex: 1,
              }}
            />

            {/* 3D Floating Neural Micro-Particles */}
            <FloatingParticlesBackground
              count={16}
              colors={["#2563EB", "#10B981", "#38BDF8", "#818CF8"]}
              opacity={0.32}
              speed={0.22}
              variant="neural"
              connectLines={true}
              maxLineDistance={80}
            />

            {/* Specular 3D Light Reflection Beam */}
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                top: 0,
                left: "8%",
                right: "8%",
                height: "2.5px",
                background:
                  "linear-gradient(90deg, transparent 0%, rgba(56, 189, 248, 0.6) 20%, rgba(255, 255, 255, 1) 50%, rgba(52, 211, 153, 0.6) 80%, transparent 100%)",
                filter: "drop-shadow(0 0 6px rgba(255,255,255,0.9))",
                zIndex: 2,
              }}
            />

            {/* 3D Dot-Matrix & Micro-Grid Texture Overlay */}
            <div
              aria-hidden="true"
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage:
                  "radial-gradient(rgba(37, 99, 235, 0.12) 1px, transparent 1px), radial-gradient(rgba(16, 185, 129, 0.08) 1px, transparent 1px)",
                backgroundSize: "20px 20px, 40px 40px",
                backgroundPosition: "0 0, 10px 10px",
                opacity: 0.65,
                pointerEvents: "none",
                maskImage:
                  "radial-gradient(ellipse 90% 80% at 50% 50%, black 40%, transparent 95%)",
                WebkitMaskImage:
                  "radial-gradient(ellipse 90% 80% at 50% 50%, black 40%, transparent 95%)",
              }}
            />

            {/* Clinical Badge */}
            <div
              style={{
                position: "relative",
                zIndex: 2,
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                border: "1px solid rgba(255, 255, 255, 0.9)",
                background: "rgba(255, 255, 255, 0.75)",
                backdropFilter: "blur(16px)",
                borderRadius: 999,
                padding: "6px 18px",
                fontSize: "0.75rem",
                color: "#1D4ED8",
                fontWeight: 800,
                letterSpacing: "0.06em",
                boxShadow:
                  "0 4px 14px rgba(37, 99, 235, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.95)",
              }}
            >
              <span
                style={{
                  background: "#2563EB",
                  color: "#fff",
                  borderRadius: 999,
                  width: 8,
                  height: 8,
                  display: "inline-block",
                  boxShadow: "0 0 8px rgba(37, 99, 235, 0.7)",
                }}
              />
              CLINICAL DERMOSCOPY AI SYSTEM
            </div>

            {/* Heading */}
            <h1
              className="syne"
              style={{
                position: "relative",
                zIndex: 2,
                fontWeight: 800,
                fontSize: "clamp(2.25rem, 5vw, 3rem)",
                color: "#111827",
                lineHeight: 1.15,
                letterSpacing: "-0.02em",
                margin: 0,
                textShadow: "0 1px 2px rgba(255,255,255,0.6)",
              }}
            >
              Skin Lesion Assessment &amp; Diagnostic Guide
            </h1>

            {/* Subtitle */}
            <p
              style={{
                position: "relative",
                zIndex: 2,
                fontSize: "0.9375rem",
                color: "#4B5563",
                lineHeight: 1.6,
                maxWidth: "540px",
                margin: 0,
              }}
            >
              Multi-architecture neural ensemble (EfficientNet-B3, Inception-V3, ConvNeXt) for dermatoscopic assessment and automated lesion segmentation.
            </p>

            {/* 3D Depth Feature Pills */}
            <div
              style={{
                position: "relative",
                zIndex: 2,
                display: "flex",
                flexWrap: "wrap",
                justifyContent: "center",
                gap: "8px",
                marginTop: "0.25rem",
              }}
            >
              {[
                { icon: "⚡", label: "3D-CA Soft Attention" },
                { icon: "🔬", label: "9 Diagnostic Classes" },
                { icon: "🛡️", label: "ISIC 2019 Calibrated" },
              ].map((item, idx) => (
                <div
                  key={idx}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                    background: "rgba(255, 255, 255, 0.7)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255, 255, 255, 0.95)",
                    padding: "5px 14px",
                    borderRadius: "999px",
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    color: "#374151",
                    boxShadow: "0 2px 8px rgba(0,0,0,0.04), inset 0 1px 1px #fff",
                  }}
                >
                  <span>{item.icon}</span>
                  <span>{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Camera/Dropzone Box */}
        <div style={{ width: "100%", maxWidth: 500, marginTop: "1rem" }}>
          <div
            onDrop={(e) => {
              e.preventDefault();
              loadFile(e.dataTransfer.files[0]);
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            style={{
              width: "100%",
              height: 260,
              background: dragging
                ? "rgba(255,255,255,0.12)"
                : "rgba(255,255,255,0.06)",
              backdropFilter: "blur(30px)",
              border: `2px dashed ${dragging ? "#3B82F6" : "rgba(255,255,255,0.2)"}`,
              borderRadius: 24,
              position: "relative",
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 0.3s ease",
            }}
          >
            {isCameraActive ? (
              <>
                <div
                  style={{
                    width: "100%",
                    height: "100%",
                    overflow: "hidden",
                    position: "relative",
                  }}
                >
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                      transform: `scaleX(-1) scale(${zoom})`,
                      transition: "transform 0.2s ease-out",
                    }}
                  />
                </div>

                {/* Zoom Controls Overlay */}
                <div
                  style={{
                    position: "absolute",
                    right: 20,
                    top: "50%",
                    transform: "translateY(-50%)",
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                    alignItems: "center",
                    background: "rgba(0,0,0,0.4)",
                    padding: "12px 8px",
                    borderRadius: 20,
                    backdropFilter: "blur(10px)",
                    zIndex: 30,
                  }}
                >
                  <button
                    onClick={() => setZoom((prev) => Math.min(prev + 0.5, 4))}
                    style={{
                      background: "none",
                      border: "none",
                      color: "#fff",
                      fontWeight: 800,
                      cursor: "pointer",
                    }}
                  >
                    +
                  </button>
                  <div
                    style={{
                      height: 100,
                      display: "flex",
                      alignItems: "center",
                    }}
                  >
                    <input
                      type="range"
                      min="1"
                      max="4"
                      step="0.1"
                      value={zoom}
                      onChange={(e) => setZoom(parseFloat(e.target.value))}
                      style={{
                        writingMode: "bt-lr",
                        appearance: "slider-vertical",
                        width: 4,
                        height: "100%",
                        cursor: "pointer",
                      }}
                    />
                  </div>
                  <button
                    onClick={() => setZoom((prev) => Math.max(prev - 0.5, 1))}
                    style={{
                      background: "none",
                      border: "none",
                      color: "#fff",
                      fontWeight: 800,
                      cursor: "pointer",
                    }}
                  >
                    −
                  </button>
                  <div
                    style={{
                      fontSize: "0.6rem",
                      color: "#fff",
                      fontWeight: 800,
                    }}
                  >
                    {zoom.toFixed(1)}x
                  </div>
                </div>

                <div
                  style={{
                    position: "absolute",
                    bottom: 20,
                    display: "flex",
                    gap: 10,
                    zIndex: 20,
                  }}
                >
                  <button
                    onClick={captureFrame}
                    style={{
                      background: "#10B981",
                      color: "#fff",
                      border: "none",
                      padding: "10px 24px",
                      borderRadius: 999,
                      fontWeight: 800,
                      cursor: "pointer",
                      boxShadow: "0 4px 15px rgba(16,185,129,0.4)",
                    }}
                  >
                    CAPTURE PHOTO
                  </button>
                  <button
                    onClick={stopCamera}
                    style={{
                      background: "rgba(0,0,0,0.5)",
                      color: "#fff",
                      border: "none",
                      padding: "10px 16px",
                      borderRadius: 999,
                      fontWeight: 800,
                      cursor: "pointer",
                    }}
                  >
                    CANCEL
                  </button>
                </div>
              </>
            ) : preview ? (
              <>
                <img
                  src={preview}
                  alt="Preview"
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
                {loading && (
                  <div
                    style={{
                      position: "absolute",
                      left: 0,
                      right: 0,
                      height: "3px",
                      background: "#3B82F6",
                      boxShadow: "0 0 15px #3B82F6",
                      animation: "scanLine 2.5s linear infinite",
                      zIndex: 10,
                    }}
                  />
                )}
                <button
                  onClick={reset}
                  style={{
                    position: "absolute",
                    top: 12,
                    right: 12,
                    background: "rgba(0,0,0,0.5)",
                    color: "#fff",
                    border: "none",
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <X size={18} />
                </button>
              </>
            ) : (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 16,
                  color: "#fff",
                }}
              >
                <Camera size={48} strokeWidth={1.5} style={{ opacity: 0.6 }} />
                <div style={{ display: "flex", gap: 12 }}>
                  <button
                    onClick={startCamera}
                    style={{
                      background: "#10B981",
                      color: "#fff",
                      border: "none",
                      padding: "10px 24px",
                      borderRadius: 999,
                      fontWeight: 700,
                      fontSize: "0.9375rem",
                      cursor: "pointer",
                      transition: "all 0.2s",
                    }}
                  >
                    Use Camera
                  </button>
                  <button
                    onClick={() => fileRef.current?.click()}
                    style={{
                      background: "rgba(255,255,255,0.1)",
                      color: "#fff",
                      border: "1px solid rgba(255,255,255,0.2)",
                      padding: "10px 24px",
                      borderRadius: 999,
                      fontWeight: 700,
                      fontSize: "0.9375rem",
                      cursor: "pointer",
                    }}
                  >
                    Upload File
                  </button>
                </div>
                <p style={{ fontSize: "0.8125rem", opacity: 0.5 }}>
                  Supported formats: JPG, PNG, DICOM
                </p>
              </div>
            )}
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>
        </div>

        {/* Patient Profile Card */}
        <div
          style={{
            position: "relative",
            zIndex: 100,
            width: "100%",
            maxWidth: 500,
            background: "rgba(255, 255, 255, 0.5)",
            backdropFilter: "blur(30px)",
            border: "1px solid rgba(0, 0, 0, 0.05)",
            borderRadius: "24px",
            padding: "1.5rem",
            display: "flex",
            flexDirection: "column",
            gap: "1.25rem",
            boxShadow: "0 20px 40px rgba(0,0,0,0.05)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "#10B981",
                }}
              />
              <span
                style={{
                  fontSize: "0.8rem",
                  color: "#111827",
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Patient Profile
              </span>
            </div>
            {(patientInfo?.name ||
              patientInfo?.gender ||
              patientInfo?.age ||
              file) && (
              <button
                onClick={reset}
                style={{
                  background: "rgba(239, 68, 68, 0.1)",
                  border: "none",
                  color: "#EF4444",
                  fontSize: "0.7rem",
                  fontWeight: 800,
                  padding: "4px 10px",
                  borderRadius: 8,
                  cursor: "pointer",
                }}
              >
                CLEAR ALL
              </button>
            )}
          </div>

          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <div style={{ position: "relative", flex: 2, minWidth: 160 }}>
              <User
                size={16}
                color="#94A3B8"
                style={{
                  position: "absolute",
                  left: 14,
                  top: "50%",
                  transform: "translateY(-50%)",
                }}
              />
              <input
                type="text"
                placeholder="Full Name"
                value={patientInfo?.name || ""}
                onChange={(e) =>
                  setPatientInfo((prev) => ({ ...prev, name: e.target.value }))
                }
                style={{
                  width: "100%",
                  height: 48,
                  background: "#fff",
                  border: "1px solid #E2E8F0",
                  color: "#111827",
                  borderRadius: 12,
                  padding: "0 16px 0 40px",
                  fontSize: "0.9rem",
                  outline: "none",
                }}
              />
            </div>
            <div style={{ position: "relative", flex: 1, minWidth: 120 }}>
              <Users
                size={16}
                color="#94A3B8"
                style={{
                  position: "absolute",
                  left: 14,
                  top: "50%",
                  transform: "translateY(-50%)",
                  zIndex: 5,
                }}
              />
              <div
                onClick={(e) => {
                  e.stopPropagation();
                  setGenderOpen(!genderOpen);
                }}
                style={{
                  height: 48,
                  background: "#fff",
                  border: "1px solid #E2E8F0",
                  color: patientInfo?.gender ? "#111827" : "#94A3B8",
                  borderRadius: 12,
                  padding: "0 16px 0 40px",
                  fontSize: "0.9rem",
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {patientInfo?.gender || "Gender"}
              </div>
              {genderOpen && (
                <div
                  style={{
                    position: "absolute",
                    top: "110%",
                    left: 0,
                    right: 0,
                    background: "#fff",
                    borderRadius: 12,
                    border: "1px solid #E2E8F0",
                    overflow: "hidden",
                    zIndex: 1000,
                    boxShadow: "0 10px 15px rgba(0,0,0,0.1)",
                  }}
                >
                  {["Male", "Female", "Other"].map((g) => (
                    <div
                      key={g}
                      onClick={() => {
                        setPatientInfo((p) => ({ ...p, gender: g }));
                        setGenderOpen(false);
                      }}
                      style={{
                        padding: "12px 16px",
                        color: "#111827",
                        fontSize: "0.9rem",
                        cursor: "pointer",
                      }}
                    >
                      {g}
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div style={{ position: "relative", width: 100 }}>
              <Hash
                size={16}
                color="#94A3B8"
                style={{
                  position: "absolute",
                  left: 14,
                  top: "50%",
                  transform: "translateY(-50%)",
                }}
              />
              <input
                type="number"
                placeholder="Age"
                value={patientInfo?.age || ""}
                onChange={(e) =>
                  setPatientInfo((prev) => ({ ...prev, age: e.target.value }))
                }
                style={{
                  width: "100%",
                  height: 48,
                  background: "#fff",
                  border: "1px solid #E2E8F0",
                  color: "#111827",
                  borderRadius: 12,
                  padding: "0 32px 0 36px",
                  fontSize: "0.9rem",
                  outline: "none",
                  textAlign: "center",
                }}
              />
              <span
                style={{
                  position: "absolute",
                  right: 12,
                  top: "50%",
                  transform: "translateY(-50%)",
                  fontSize: "0.6rem",
                  color: "#059669",
                  fontWeight: 800,
                }}
              >
                YRS
              </span>
            </div>
          </div>
          {!validation.isValid && validation.errors?.age?._errors?.length > 0 && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                color: "#EF4444",
                fontSize: "0.75rem",
                fontWeight: 600,
                padding: "4px 8px",
                background: "rgba(239,68,68,0.08)",
                borderRadius: "8px",
              }}
            >
              <AlertCircle size={14} />
              {validation.errors.age._errors[0]}
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
          <button
            disabled={!file || loading}
            onClick={() => onAnalyze(file)}
            style={{
              background: "#2563EB",
              color: "#fff",
              border: "none",
              padding: "14px 32px",
              borderRadius: 999,
              fontWeight: 700,
              fontSize: "0.9375rem",
              cursor: !file || loading ? "not-allowed" : "pointer",
              opacity: !file || loading ? 0.6 : 1,
              boxShadow: "0 10px 20px rgba(37, 99, 235, 0.3)",
              transition: "all 0.3s",
            }}
          >
            {loading ? "Analyzing..." : "Run Diagnostics"}
          </button>
        </div>

        {/* Clinical Disclaimer Banner */}
        <div
          style={{
            marginTop: "1.25rem",
            maxWidth: 500,
            width: "100%",
            background: "rgba(255, 255, 255, 0.85)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(234, 179, 8, 0.4)",
            borderRadius: "16px",
            padding: "0.85rem 1.25rem",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            fontSize: "0.78rem",
            color: "#475569",
            lineHeight: 1.4,
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.03)",
          }}
        >
          <span style={{ fontSize: "1.1rem" }}>⚖️</span>
          <div>
            <strong style={{ color: "#1E293B" }}>Clinical Disclaimer:</strong> For educational & research reference only. Always consult a certified dermatologist for medical diagnosis.
          </div>
        </div>

        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          style={{ display: "none" }}
          onChange={(e) => loadFile(e.target.files[0])}
        />
      </div>

      {/* Model Badges */}
      <div
        style={{
          display: "flex",
          gap: "1rem",
          flexWrap: "wrap",
          justifyContent: "center",
          padding: "2rem",
          zIndex: 2,
        }}
      >
        {MODEL_BADGES.map((m) => (
          <div
            key={m.name}
            style={{
              background: "rgba(255,255,255,0.4)",
              backdropFilter: "blur(10px)",
              border: "1px solid rgba(0,0,0,0.1)",
              borderRadius: 16,
              padding: "1rem",
              display: "flex",
              alignItems: "center",
              gap: 12,
              minWidth: 200,
            }}
          >
            <div style={{ fontSize: "1.5rem" }}>{m.icon}</div>
            <div>
              <div
                style={{
                  color: "#111827",
                  fontWeight: 700,
                  fontSize: "0.9rem",
                }}
              >
                {m.name}
              </div>
              <div
                style={{
                  color: "#059669",
                  fontSize: "0.7rem",
                  fontWeight: 800,
                }}
              >
                ACTIVE MODEL
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
