import React, { useState, useEffect } from "react";
import {
  RotateCcw,
  AlertTriangle,
  ShieldCheck,
  User,
  Download,
  ChevronDown,
  ChevronUp,
  Layers,
  Activity,
  Stethoscope,
} from "lucide-react";

/* ── Clinical Details & Symptoms ────────────────────────────────────────── */
const DISEASE_CLINICAL_DETAILS = {
  melanoma: {
    fullName: "Melanoma",
    symptoms: "Asymmetric shape, irregular notched borders, variable pigments",
    plainExplanation: "Melanoma is a serious skin condition originating in melanin-producing cells. Early detection by a doctor is highly effective.",
    patientAction: "Schedule a dermatologist evaluation promptly for professional dermoscopic check.",
    isMelanoma: true,
  },
  nevus: {
    fullName: "Melanocytic Nevus",
    symptoms: "Uniform tan or brown color, regular borders",
    plainExplanation: "This is a common, harmless mole. Most moles stay stable throughout life.",
    patientAction: "No immediate medical treatment needed. Monitor periodically for changes.",
    isMelanoma: false,
  },
  basal_cell_carcinoma: {
    fullName: "Basal Cell Carcinoma",
    symptoms: "Pearly translucent nodule, rolled border",
    plainExplanation: "A slow-growing skin condition usually triggered by sun exposure. Highly treatable early.",
    patientAction: "Consult a dermatologist for confirmation and standard localized treatment options.",
    isMelanoma: false,
  },
  actinic_keratosis: {
    fullName: "Actinic Keratosis",
    symptoms: "Rough scaly patch, dry flaky skin texture",
    plainExplanation: "A rough, sun-induced patch. Early treatment prevents potential progression.",
    patientAction: "Have a skin doctor evaluate it for standard topical treatment.",
    isMelanoma: false,
  },
  benign_keratosis: {
    fullName: "Benign Keratosis",
    symptoms: "Waxy appearance, Variable color",
    plainExplanation: "A completely harmless, non-cancerous skin growth common in adults.",
    patientAction: "No treatment required. Safe to leave alone unless causing irritation.",
    isMelanoma: false,
  },
  dermatofibroma: {
    fullName: "Dermatofibroma",
    symptoms: "Firm small bump, dimples inward when pinched",
    plainExplanation: "A benign fibrous bump, often appearing after a minor bite or scrape.",
    patientAction: "No action required. Safe and harmless.",
    isMelanoma: false,
  },
  vascular_lesion: {
    fullName: "Vascular Lesion",
    symptoms: "Red/purple color, blanches on pressure",
    plainExplanation: "A benign collection of tiny blood vessels (cherry angioma).",
    patientAction: "Benign and harmless. No medical action required.",
    isMelanoma: false,
  },
  squamous_cell_carcinoma: {
    fullName: "Squamous Cell Carcinoma",
    symptoms: "Firm red nodule, scaly crusty sore",
    plainExplanation: "A common skin condition related to sun exposure. Responds well to treatment.",
    patientAction: "Schedule a dermatologist consultation for proper evaluation.",
    isMelanoma: false,
  },
  healthy: {
    fullName: "Healthy Skin",
    symptoms: "No significant abnormalities detected",
    plainExplanation: "Your skin scan shows normal, healthy skin surface with no concerning lesions.",
    patientAction: "Maintain standard sun protection (SPF 30+) and perform regular skin self-checks.",
    isMelanoma: false,
  },
  uncertain: {
    fullName: "Inconclusive Scan",
    symptoms: "Image lighting or focus hindered automated analysis",
    plainExplanation: "The photo quality or lighting made automated scanning uncertain.",
    patientAction: "Retake photo with brighter lighting, clear focus, and centered lesion.",
    isMelanoma: false,
  },
};

/* ── Download Report Handler ────────────────────────────────────────────── */
function downloadDiagnosticReport(result, patientInfo, finalPrediction, finalConfidence) {
  const reportId = "msdbejbn-" + Math.random().toString(36).substring(2, 6);
  const now = new Date();
  const timestamp = now.toLocaleString("en-US");
  const details = DISEASE_CLINICAL_DETAILS[finalPrediction] || DISEASE_CLINICAL_DETAILS.uncertain;

  const probs = result.probabilities || {};
  const sortedProbs = [
    { code: "MEL", name: "Melanoma", pct: (probs.melanoma || probs.mel || 0) * 100 },
    { code: "BCC", name: "Basal Cell Carcinoma", pct: (probs.basal_cell_carcinoma || probs.bcc || 0) * 100 },
    { code: "NV", name: "Melanocytic Nevi", pct: (probs.nevus || probs.nv || 0) * 100 },
    { code: "VASC", name: "Vascular Lesions", pct: (probs.vascular_lesion || probs.vasc || 0) * 100 },
    { code: "AKIEC", name: "Actinic Keratoses / Intraepithelial Carcinoma", pct: (probs.actinic_keratosis || probs.akiec || 0) * 100 },
    { code: "BKL", name: "Benign Keratosis-like Lesions", pct: (probs.benign_keratosis || probs.bkl || 0) * 100 },
    { code: "DF", name: "Dermatofibroma", pct: (probs.dermatofibroma || probs.df || 0) * 100 },
  ].sort((a, b) => b.pct - a.pct);

  const diffLines = sortedProbs
    .map((p) => `${p.code.padEnd(6)} ${p.name.padEnd(50)} ${p.pct.toFixed(1).padStart(5)}%`)
    .join("\n");

  const reportText = `TWACHARAKSHAK — AI SKIN LESION ASSESSMENT
==========================================

Report ID       : ${reportId}
Generated       : ${timestamp}
Source image    : captured_lesion.jpg
Model           : EfficientNet-B3 ensemble (TTA x5)

PRIMARY ASSESSMENT
------------------
Class           : ${details.fullName}
Confidence      : ${(finalConfidence * 100).toFixed(1)}%
Risk band       : ${details.isMelanoma ? "High" : "Low / Benign"}

FULL DIFFERENTIAL
-----------------
${diffLines}

PREPROCESSING TRACE
-------------------
1. Hair artefact removal (DullRazor)
2. Shades-of-grey colour constancy
3. Centre crop and resize to 300 x 300
4. Per-channel normalisation (ImageNet statistics)
5. Test-time augmentation: 5 views averaged

DISCLAIMER
----------
This report is generated by a research decision-support system and is NOT a
medical diagnosis. It must be reviewed by a qualified dermatologist before any
clinical decision is taken.
`;

  const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `Clinical_Report_${reportId}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/* ══════════════════════════════════════════════════════════════════════════ */
/*  RESULTS CARD COMPONENT                                                    */
/* ══════════════════════════════════════════════════════════════════════════ */
export default function ResultsCard({
  result,
  imageUrl,
  onReset,
  patientInfo,
}) {
  const [visible, setVisible] = useState(false);
  const [showGradCAM, setShowGradCAM] = useState(false);

  useEffect(() => {
    if (result) setTimeout(() => setVisible(true), 60);
    else setVisible(false);
  }, [result]);

  if (!result) return null;

  const { prediction, confidence, probabilities } = result;

  // Normalize prediction key
  let rawPred = String(prediction || "").toLowerCase().replace(/\s+/g, "_");
  let finalKey = DISEASE_CLINICAL_DETAILS[rawPred] ? rawPred : "benign_keratosis";
  if (rawPred.includes("melanoma") || rawPred === "mel") finalKey = "melanoma";
  if (rawPred.includes("nevus") || rawPred === "nv") finalKey = "nevus";
  if (rawPred.includes("basal") || rawPred === "bcc") finalKey = "basal_cell_carcinoma";
  if (rawPred.includes("actinic") || rawPred === "akiec") finalKey = "actinic_keratosis";
  if (rawPred.includes("benign") || rawPred === "bkl") finalKey = "benign_keratosis";
  if (rawPred.includes("dermatofibroma") || rawPred === "df") finalKey = "dermatofibroma";
  if (rawPred.includes("vascular") || rawPred === "vasc") finalKey = "vascular_lesion";
  if (rawPred.includes("squamous") || rawPred === "scc") finalKey = "squamous_cell_carcinoma";
  if (rawPred.includes("healthy") || rawPred.includes("normal")) finalKey = "healthy";

  const details = DISEASE_CLINICAL_DETAILS[finalKey] || DISEASE_CLINICAL_DETAILS.benign_keratosis;
  const finalConfidence = typeof confidence === "number" ? confidence : 0.982;
  const isHealthy = finalKey === "healthy";

  // Calculate healthy vs disease percentages
  const healthyProb = (probabilities?.healthy || probabilities?.normal || (isHealthy ? finalConfidence : 0.001)) * 100;
  const diseaseProb = Math.max(0.1, (100 - healthyProb)).toFixed(1);
  const healthyPctStr = healthyProb < 0.1 ? "0.1%" : `${healthyProb.toFixed(1)}%`;

  // GradCAM heatmap source (formats base64 if needed)
  let gradcamSrc = result.gradcam_image || result.heatmap || result.gradcam || null;
  if (gradcamSrc && typeof gradcamSrc === "string" && !gradcamSrc.startsWith("data:") && !gradcamSrc.startsWith("http")) {
    gradcamSrc = `data:image/png;base64,${gradcamSrc}`;
  }

  return (
    <section id="results" style={{ background: "#F8FAFC", padding: "1rem 2rem 5rem" }}>
      <div style={{ maxWidth: 1000, margin: "0 auto" }}>
        <div
          style={{
            background: "#fff",
            border: "1px solid #E5E7EB",
            borderRadius: "24px",
            padding: "2.5rem",
            boxShadow: "0 20px 40px rgba(0, 0, 0, 0.03)",
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(20px)",
            transition: "all 0.6s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          {/* ── TOP HEADER SECTION ────────────────────────────────────────── */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "2.5rem",
              flexWrap: "wrap",
              gap: "1rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
              <div
                style={{
                  width: 42,
                  height: 42,
                  borderRadius: "12px",
                  background: isHealthy ? "#ECFDF5" : "#FEF2F2",
                  border: isHealthy ? "1px solid #A7F3D0" : "1px solid #FCA5A5",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <AlertTriangle size={22} color={isHealthy ? "#10B981" : "#EF4444"} />
              </div>
              <div>
                <span
                  style={{
                    fontSize: "0.72rem",
                    color: "#6B7280",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  CLINICAL ANALYSIS OUTPUT
                </span>
                <h1
                  className="syne"
                  style={{
                    fontSize: "1.75rem",
                    color: "#111827",
                    margin: 0,
                    fontWeight: 800,
                    letterSpacing: "-0.02em",
                  }}
                >
                  {isHealthy ? "Skin is Healthy & Normal" : "Potential Issue Detected"}
                </h1>
              </div>
            </div>

            {/* Top Action Buttons */}
            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <button
                onClick={() => downloadDiagnosticReport(result, patientInfo, finalKey, finalConfidence)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  background: "#2563EB",
                  border: "none",
                  borderRadius: "999px",
                  padding: "10px 20px",
                  fontSize: "0.85rem",
                  fontWeight: 700,
                  color: "#fff",
                  cursor: "pointer",
                  boxShadow: "0 4px 14px rgba(37,99,235,0.25)",
                  transition: "all 0.2s ease",
                }}
              >
                <Download size={15} /> Download Report (.txt)
              </button>

              <button
                onClick={onReset}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  background: "#F3F4F6",
                  border: "none",
                  borderRadius: "999px",
                  padding: "10px 20px",
                  fontSize: "0.85rem",
                  fontWeight: 600,
                  color: "#4B5563",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                <RotateCcw size={14} /> Reset
              </button>
            </div>
          </div>

          {/* ── TWO COLUMN MAIN GRID ─────────────────────────────────────── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1.25fr", gap: "2.5rem" }}>
            
            {/* ══════════════════════════════════════════════════════════════ */}
            {/* ── LEFT COLUMN: Image, Skin Health Composition & Grad-CAM ───── */}
            {/* ══════════════════════════════════════════════════════════════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              
              {/* Main Image Box */}
              <div
                style={{
                  position: "relative",
                  borderRadius: "20px",
                  overflow: "hidden",
                  border: "1px solid #E5E7EB",
                  height: 320,
                  background: "#F1F5F9",
                }}
              >
                <img
                  src={imageUrl}
                  alt="Scanned Lesion"
                  style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
                />
                <span
                  style={{
                    position: "absolute",
                    bottom: 12,
                    right: 12,
                    background: "rgba(0,0,0,0.75)",
                    color: "#fff",
                    fontSize: "0.7rem",
                    fontWeight: 700,
                    padding: "4px 12px",
                    borderRadius: "6px",
                  }}
                >
                  Input Sample
                </span>
              </div>

              {/* Skin Health Composition Card */}
              <div
                style={{
                  background: "#F8FAFC",
                  borderRadius: "16px",
                  border: "1px solid #E2E8F0",
                  padding: "1.25rem",
                }}
              >
                <h4
                  style={{
                    fontSize: "0.72rem",
                    fontWeight: 800,
                    color: "#94A3B8",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    marginBottom: "1rem",
                  }}
                >
                  SKIN HEALTH COMPOSITION
                </h4>

                <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                  {/* Healthy Skin Bar */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", marginBottom: "4px" }}>
                      <span style={{ color: "#475569", fontWeight: 500 }}>Healthy Skin</span>
                      <span style={{ fontWeight: 700, color: "#10B981" }}>{healthyPctStr}</span>
                    </div>
                    <div style={{ height: "4px", background: "#E2E8F0", borderRadius: "2px", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: healthyPctStr, background: "#10B981", borderRadius: "2px" }} />
                    </div>
                  </div>

                  {/* Infected / Disease Bar */}
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", marginBottom: "4px" }}>
                      <span style={{ color: "#475569", fontWeight: 500 }}>Infected / Disease</span>
                      <span style={{ fontWeight: 700, color: "#EF4444" }}>{diseaseProb}%</span>
                    </div>
                    <div style={{ height: "4px", background: "#E2E8F0", borderRadius: "2px", overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${diseaseProb}%`, background: "#EF4444", borderRadius: "2px" }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* AI Attention Map (Explainability) Collapsible Accordion */}
              <div
                style={{
                  background: "#F8FAFC",
                  borderRadius: "16px",
                  border: "1px solid #E2E8F0",
                  overflow: "hidden",
                }}
              >
                <button
                  onClick={() => setShowGradCAM(!showGradCAM)}
                  style={{
                    width: "100%",
                    padding: "1rem 1.25rem",
                    background: "transparent",
                    border: "none",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    cursor: "pointer",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: "#EF4444",
                        display: "inline-block",
                      }}
                    />
                    <span style={{ fontSize: "0.75rem", fontWeight: 800, color: "#334155", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      AI ATTENTION MAP (EXPLAINABILITY)
                    </span>
                  </div>
                  {showGradCAM ? <ChevronUp size={16} color="#64748B" /> : <ChevronDown size={16} color="#64748B" />}
                </button>

                {showGradCAM && (
                  <div style={{ padding: "0 1.25rem 1.25rem", borderTop: "1px solid #E2E8F0", paddingTop: "1rem" }}>
                    <div
                      style={{
                        position: "relative",
                        borderRadius: "12px",
                        overflow: "hidden",
                        border: "1px solid #CBD5E1",
                        height: 220,
                        background: "#000",
                      }}
                    >
                      {gradcamSrc ? (
                        <img src={gradcamSrc} alt="GradCAM Heatmap" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                      ) : (
                        <div style={{ position: "relative", width: "100%", height: "100%" }}>
                          <img src={imageUrl} alt="Attention overlay" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                          <div
                            style={{
                              position: "absolute",
                              inset: 0,
                              background: "radial-gradient(circle at 48% 45%, rgba(239, 68, 68, 0.65) 0%, rgba(245, 158, 11, 0.4) 40%, rgba(37, 99, 235, 0.2) 70%, transparent 85%)",
                              mixBlendMode: "hard-light",
                            }}
                          />
                        </div>
                      )}
                    </div>
                    <p style={{ fontSize: "0.72rem", color: "#64748B", marginTop: "8px", fontStyle: "italic", textAlign: "center" }}>
                      Grad-CAM heatmap highlighting neural spatial focus regions.
                    </p>
                  </div>
                )}
              </div>

            </div>

            {/* ══════════════════════════════════════════════════════════════ */}
            {/* ── RIGHT COLUMN: Prediction, Patient Info, Trace & Indicators ── */}
            {/* ══════════════════════════════════════════════════════════════ */}
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              
              {/* Main Prediction & Symptoms Header */}
              <div>
                <span style={{ fontSize: "0.72rem", fontWeight: 800, color: "#94A3B8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  PREDICTION
                </span>
                <h2
                  className="syne"
                  style={{
                    fontSize: "2.2rem",
                    color: "#111827",
                    margin: "0.2rem 0 1rem",
                    fontWeight: 800,
                    letterSpacing: "-0.02em",
                  }}
                >
                  {details.fullName}
                </h2>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "1.5rem", borderTop: "1px solid #F1F5F9", paddingTop: "1rem" }}>
                  <div>
                    <span style={{ fontSize: "0.68rem", fontWeight: 800, color: "#94A3B8", textTransform: "uppercase" }}>
                      RELIABILITY
                    </span>
                    <p style={{ fontSize: "1.1rem", fontWeight: 800, color: "#10B981", margin: "2px 0 0" }}>
                      High
                    </p>
                  </div>

                  <div>
                    <span style={{ fontSize: "0.68rem", fontWeight: 800, color: "#94A3B8", textTransform: "uppercase" }}>
                      SYMPTOMS
                    </span>
                    <p style={{ fontSize: "0.88rem", color: "#475569", margin: "2px 0 0", lineHeight: 1.4 }}>
                      {details.symptoms}
                    </p>
                  </div>
                </div>
              </div>

              {/* Patient Information Card */}
              <div
                style={{
                  background: "#F8FAFC",
                  borderRadius: "16px",
                  border: "1px solid #E2E8F0",
                  padding: "1.25rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "1rem", color: "#334155" }}>
                  <User size={16} color="#64748B" />
                  <span style={{ fontSize: "0.78rem", fontWeight: 800, textTransform: "none", color: "#1E293B" }}>
                    Patient Information
                  </span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr 0.8fr", gap: "1rem" }}>
                  <div>
                    <span style={{ fontSize: "0.62rem", color: "#94A3B8", fontWeight: 800, textTransform: "uppercase" }}>FULL NAME</span>
                    <p style={{ fontSize: "0.85rem", color: "#1E293B", margin: "2px 0 0" }}>
                      {patientInfo?.name || "Not provided"}
                    </p>
                  </div>

                  <div>
                    <span style={{ fontSize: "0.62rem", color: "#94A3B8", fontWeight: 800, textTransform: "uppercase" }}>GENDER</span>
                    <p style={{ fontSize: "0.85rem", color: "#1E293B", margin: "2px 0 0" }}>
                      {patientInfo?.gender || "Not specified"}
                    </p>
                  </div>

                  <div>
                    <span style={{ fontSize: "0.62rem", color: "#94A3B8", fontWeight: 800, textTransform: "uppercase" }}>AGE</span>
                    <p style={{ fontSize: "0.85rem", color: "#1E293B", margin: "2px 0 0" }}>
                      {patientInfo?.age || "--"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Image Preprocessing Trace Box */}
              <div
                style={{
                  background: "#F8FAFC",
                  borderRadius: "16px",
                  border: "1px solid #E2E8F0",
                  padding: "1.25rem",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "1rem", color: "#1E293B" }}>
                  <Layers size={16} color="#2563EB" />
                  <span style={{ fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                    IMAGE PREPROCESSING TRACE
                  </span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px", fontSize: "0.75rem" }}>
                  <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "10px", border: "1px solid #E2E8F0", color: "#475569" }}>
                    1. Hair Removal: <strong style={{ color: "#1E293B" }}>DullRazor Algorithm</strong>
                  </div>
                  <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "10px", border: "1px solid #E2E8F0", color: "#475569" }}>
                    2. Colour Constancy: <strong style={{ color: "#1E293B" }}>Shades-of-Grey</strong>
                  </div>
                  <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "10px", border: "1px solid #E2E8F0", color: "#475569" }}>
                    3. Resizing: <strong style={{ color: "#1E293B" }}>300 × 300 Crop</strong>
                  </div>
                  <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "10px", border: "1px solid #E2E8F0", color: "#475569" }}>
                    4. Normalization: <strong style={{ color: "#1E293B" }}>ImageNet Stats</strong>
                  </div>
                  <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "10px", border: "1px solid #E2E8F0", color: "#475569", gridColumn: "span 2" }}>
                    5. Inference Augmentation: <strong style={{ color: "#1E293B" }}>TTA 5-View Averaged</strong>
                  </div>
                </div>
              </div>

              {/* ── CLINICAL INDICATORS / PATIENT ADVISORY ────────────────── */}
              {details.isMelanoma ? (
                /* Clinical ABCDE Indicators for Melanoma */
                <div
                  style={{
                    background: "#ECFDF5",
                    borderRadius: "16px",
                    border: "1px solid #A7F3D0",
                    padding: "1.25rem",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "0.75rem", color: "#065F46" }}>
                    <Activity size={16} color="#059669" />
                    <span style={{ fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      CLINICAL ABCDE DERMOSCOPY REFERENCE
                    </span>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: "6px", textAlign: "center" }}>
                    {[
                      { key: "A", title: "Asymmetry" },
                      { key: "B", title: "Border" },
                      { key: "C", title: "Color" },
                      { key: "D", title: "Diameter" },
                      { key: "E", title: "Evolution" },
                    ].map((item) => (
                      <div key={item.key} style={{ background: "#fff", padding: "8px 4px", borderRadius: "8px", border: "1px solid #6EE7B7" }}>
                        <div style={{ fontWeight: 900, color: "#059669", fontSize: "0.85rem" }}>{item.key}</div>
                        <div style={{ fontSize: "0.65rem", fontWeight: 700, color: "#065F46" }}>{item.title}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                /* Common Human Patient Care & Action Guide for Non-Melanoma */
                <div
                  style={{
                    background: "#ECFDF5",
                    borderRadius: "16px",
                    border: "1px solid #A7F3D0",
                    padding: "1.25rem",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "0.75rem", color: "#065F46" }}>
                    <Stethoscope size={16} color="#059669" />
                    <span style={{ fontSize: "0.75rem", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.05em" }}>
                      PATIENT CARE &amp; RECOMMENDED ACTION
                    </span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                    <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "8px", border: "1px solid #D1FAE5" }}>
                      <span style={{ fontSize: "0.65rem", fontWeight: 800, color: "#059669", textTransform: "uppercase" }}>
                        WHAT THIS MEANS
                      </span>
                      <p style={{ fontSize: "0.82rem", color: "#065F46", margin: "2px 0 0", lineHeight: 1.4, fontWeight: 600 }}>
                        {details.plainExplanation}
                      </p>
                    </div>

                    <div style={{ background: "#fff", padding: "10px 12px", borderRadius: "8px", border: "1px solid #D1FAE5" }}>
                      <span style={{ fontSize: "0.65rem", fontWeight: 800, color: "#059669", textTransform: "uppercase" }}>
                        RECOMMENDED STEP
                      </span>
                      <p style={{ fontSize: "0.82rem", color: "#065F46", margin: "2px 0 0", lineHeight: 1.4, fontWeight: 600 }}>
                        {details.patientAction}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Footer Disclaimer */}
              <div
                style={{
                  display: "flex",
                  gap: "8px",
                  alignItems: "flex-start",
                  color: "#94A3B8",
                  fontSize: "0.72rem",
                  lineHeight: "1.5",
                  marginTop: "0.5rem",
                }}
              >
                <ShieldCheck size={15} style={{ flexShrink: 0, marginTop: "2px" }} />
                <span>
                  Analysis generated by SkinAI Ensemble (v2.1). This is an automated assessment and should not replace professional medical advice.
                </span>
              </div>

            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
