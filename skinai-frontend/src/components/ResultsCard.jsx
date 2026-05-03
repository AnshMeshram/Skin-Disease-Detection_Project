import { useState, useEffect } from "react";
import { RotateCcw, AlertTriangle, ShieldCheck, User } from "lucide-react";

const DISEASE_INFO = {
  melanoma: { symptoms: "Patches, Itchy Skin, Fever", is_healthy: false },
  nevus: {
    symptoms: "Uniform pigmentation, Regular borders",
    is_healthy: false,
  },
  basal_cell_carcinoma: {
    symptoms: "Pearly nodule, Rolled border, Telangiectasia",
    is_healthy: false,
  },
  actinic_keratosis: {
    symptoms: "Rough scaly patch, Redness, Itching",
    is_healthy: false,
  },
  benign_keratosis: {
    symptoms: "Waxy appearance, Variable color",
    is_healthy: false,
  },
  dermatofibroma: {
    symptoms: "Dimple sign, Firm nodule, Hyperpigmentation",
    is_healthy: false,
  },
  vascular_lesion: {
    symptoms: "Red/purple color, Blanches on pressure",
    is_healthy: false,
  },
  squamous_cell_carcinoma: {
    symptoms: "Ulceration, Hyperkeratosis, Induration",
    is_healthy: false,
  },
  healthy: {
    symptoms: "No significant abnormalities detected",
    is_healthy: true,
  },
};

function cap(str) {
  if (!str) return "—";
  return str.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function ResultsCard({
  result,
  imageUrl,
  onReset,
  patientInfo,
}) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (result) setTimeout(() => setVisible(true), 60);
    else setVisible(false);
  }, [result]);

  if (!result) return null;

  const { prediction, confidence, probabilities, is_uncertain } = result;

  // --- Lesion Verification Logic (v5.1 Safety Lock) ---
  let finalPrediction = prediction;
  let finalConfidence = confidence || 0;
  let isOverridden = false;

  const healthyProb =
    probabilities?.["Healthy Skin"] ||
    probabilities?.["Healthy"] ||
    probabilities?.["healthy"] ||
    probabilities?.["Normal"] ||
    0;

  // Rule 1: FIXED Safety Lock - Protect Against False Negatives (infected skin shown as healthy)
  // If prediction is HEALTHY but confidence is LOW, and there's significant disease probability, flag as uncertain
  if (finalPrediction === "healthy" && finalConfidence < 0.96) {
    const maxDiseaseProb = Math.max(
      ...Object.entries(probabilities || {})
        .filter(
          ([k]) =>
            k !== "Healthy" &&
            k !== "Healthy Skin" &&
            k !== "healthy" &&
            k !== "Normal",
        )
        .map(([_, v]) => v || 0),
    );

    if (maxDiseaseProb > 0.04) {
      // Force uncertainty flag instead of accepting low-confidence healthy
      isOverridden = true;
    }
  }

  // Rule 2: FIXED - For disease predictions, require HIGH confidence to change them to healthy
  // We NEVER want to downgrade a disease prediction to healthy unless we're very sure
  if (finalPrediction !== "healthy" && finalConfidence >= 0.75) {
    // Disease prediction with decent confidence - KEEP IT, don't override to healthy
    isOverridden = false;
  } else if (
    finalPrediction !== "healthy" &&
    finalConfidence < 0.75 &&
    healthyProb > 0.75
  ) {
    // Only if healthy probability is DOMINANT, consider flagging as uncertain
    isOverridden = true;
  }

  const info = DISEASE_INFO[finalPrediction] || {
    symptoms: "N/A",
    is_healthy: false,
  };
  const confPct = (finalConfidence * 100).toFixed(2);
  const displayHealthyResult = info.is_healthy;

  let reliability = "High";
  let reliabilityColor = "#10B981";
  if (!isOverridden) {
    if (finalConfidence < 0.75) {
      reliability = "Medium";
      reliabilityColor = "#F59E0B";
    }
    if (finalConfidence < 0.55 || is_uncertain) {
      reliability = "Low";
      reliabilityColor = "#EF4444";
    }
  }

  const displayHealthyProb =
    finalPrediction === "healthy" ? finalConfidence : healthyProb;
  const displayDiseaseProb = Math.max(0, 1 - displayHealthyProb);

  return (
    <section
      id="results"
      style={{ background: "#F8FAFC", padding: "0 2rem 3rem" }}
    >
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div
          style={{
            background: "#fff",
            border: "1px solid #e5e7eb",
            borderRadius: "24px",
            padding: "2.5rem",
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.1)",
            opacity: visible ? 1 : 0,
            transform: visible ? "translateY(0)" : "translateY(20px)",
            transition: "all 0.6s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          {/* Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "2rem",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div
                style={{
                  padding: "8px",
                  background: displayHealthyResult ? "#ECFDF5" : "#FEF2F2",
                  borderRadius: "12px",
                }}
              >
                {displayHealthyResult ? (
                  <ShieldCheck color="#10B981" />
                ) : (
                  <AlertTriangle color="#EF4444" />
                )}
              </div>
              <div>
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "#6B7280",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  Clinical Analysis Output
                </span>
                <h2
                  className="syne"
                  style={{ fontSize: "1.25rem", color: "#111827" }}
                >
                  {displayHealthyResult
                    ? "Skin is Healthy"
                    : "Potential Issue Detected"}
                </h2>
              </div>
            </div>
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
                transition: "all 0.2s",
              }}
            >
              <RotateCcw size={14} /> Reset
            </button>
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1.5fr",
              gap: "2.5rem",
            }}
          >
            {/* Left: Image & Quick Stats */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.5rem",
              }}
            >
              <div
                style={{
                  position: "relative",
                  borderRadius: "16px",
                  overflow: "hidden",
                  border: "1px solid #E5E7EB",
                }}
              >
                <img
                  src={imageUrl}
                  alt="Analyzed"
                  style={{
                    width: "100%",
                    aspectRatio: "1",
                    objectFit: "cover",
                  }}
                />
                <div
                  style={{
                    position: "absolute",
                    bottom: "12px",
                    right: "12px",
                    background: "rgba(0,0,0,0.6)",
                    color: "#fff",
                    padding: "4px 10px",
                    borderRadius: "6px",
                    fontSize: "0.7rem",
                    backdropFilter: "blur(4px)",
                  }}
                >
                  Input Sample
                </div>
              </div>

              {/* Disease vs Healthy Graph */}
              <div
                style={{
                  background: "#F9FAFB",
                  padding: "1.25rem",
                  borderRadius: "16px",
                  border: "1px solid #E5E7EB",
                }}
              >
                <p
                  style={{
                    fontSize: "0.7rem",
                    fontWeight: 700,
                    color: "#9CA3AF",
                    textTransform: "uppercase",
                    marginBottom: "1rem",
                  }}
                >
                  Skin Health Composition
                </p>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "12px",
                  }}
                >
                  <div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.8rem",
                        marginBottom: "4px",
                      }}
                    >
                      <span style={{ color: "#4B5563" }}>Healthy Skin</span>
                      <span style={{ fontWeight: 600, color: "#10B981" }}>
                        {(displayHealthyProb * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div
                      style={{
                        height: "6px",
                        background: "#E5E7EB",
                        borderRadius: "3px",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${displayHealthyProb * 100}%`,
                          background: "#10B981",
                          borderRadius: "3px",
                          transition: "width 1s ease",
                        }}
                      />
                    </div>
                  </div>
                  <div>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        fontSize: "0.8rem",
                        marginBottom: "4px",
                      }}
                    >
                      <span style={{ color: "#4B5563" }}>
                        Infected / Disease
                      </span>
                      <span style={{ fontWeight: 600, color: "#EF4444" }}>
                        {(displayDiseaseProb * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div
                      style={{
                        height: "6px",
                        background: "#E5E7EB",
                        borderRadius: "3px",
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${displayDiseaseProb * 100}%`,
                          background: "#EF4444",
                          borderRadius: "3px",
                          transition: "width 1s ease",
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Detailed Info */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.5rem",
              }}
            >
              <div
                style={{
                  borderBottom: "1px solid #F3F4F6",
                  paddingBottom: "1rem",
                }}
              >
                <span
                  style={{
                    fontSize: "0.75rem",
                    color: "#9CA3AF",
                    fontWeight: 600,
                  }}
                >
                  PREDICTION
                </span>
                <h3
                  className="syne"
                  style={{
                    fontSize: "1.75rem",
                    color: displayHealthyResult ? "#10B981" : "#111827",
                    marginTop: "4px",
                  }}
                >
                  {cap(finalPrediction)}
                </h3>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "1.5rem",
                }}
              >
                <div>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      color: "#9CA3AF",
                      fontWeight: 600,
                    }}
                  >
                    RELIABILITY
                  </span>
                  <p
                    style={{
                      fontSize: "1.1rem",
                      fontWeight: 700,
                      color: reliabilityColor,
                    }}
                  >
                    {reliability}
                  </p>
                </div>
                <div>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      color: "#9CA3AF",
                      fontWeight: 600,
                    }}
                  >
                    SYMPTOMS
                  </span>
                  <p
                    style={{
                      fontSize: "0.9rem",
                      color: "#4B5563",
                      lineHeight: "1.4",
                    }}
                  >
                    {info.symptoms}
                  </p>
                </div>
              </div>

              <div
                style={{
                  background: "#F9FAFB",
                  padding: "1.5rem",
                  borderRadius: "16px",
                  border: "1px solid #E5E7EB",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "1rem",
                    color: "#4B5563",
                  }}
                >
                  <User size={16} />
                  <span style={{ fontSize: "0.8rem", fontWeight: 600 }}>
                    Patient Information
                  </span>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: "1rem",
                  }}
                >
                  <div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        color: "#9CA3AF",
                        fontWeight: 600,
                      }}
                    >
                      FULL NAME
                    </span>
                    <p style={{ fontSize: "0.85rem", color: "#111827" }}>
                      {patientInfo?.name || "Not provided"}
                    </p>
                  </div>
                  <div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        color: "#9CA3AF",
                        fontWeight: 600,
                      }}
                    >
                      GENDER
                    </span>
                    <p style={{ fontSize: "0.85rem", color: "#111827" }}>
                      {patientInfo?.gender || "Not specified"}
                    </p>
                  </div>
                  <div>
                    <span
                      style={{
                        fontSize: "0.65rem",
                        color: "#9CA3AF",
                        fontWeight: 600,
                      }}
                    >
                      AGE
                    </span>
                    <p style={{ fontSize: "0.85rem", color: "#111827" }}>
                      {patientInfo?.age || "--"}
                    </p>
                  </div>
                </div>
              </div>

              {finalConfidence < 0.75 && (
                <div
                  style={{
                    display: "flex",
                    gap: "10px",
                    padding: "1rem",
                    background: "#FFFBEB",
                    borderRadius: "12px",
                    border: "1px solid #FDE68A",
                  }}
                >
                  <AlertTriangle
                    size={18}
                    color="#D97706"
                    style={{ flexShrink: 0 }}
                  />
                  <div>
                    <p
                      style={{
                        fontSize: "0.85rem",
                        fontWeight: 700,
                        color: "#92400E",
                      }}
                    >
                      Low Confidence Notice
                    </p>
                    <p style={{ fontSize: "0.75rem", color: "#B45309" }}>
                      This detection has a reliability rating of {reliability}.
                      For better results, ensure the lesion is centered,
                      well-lit, and try zooming in (2x).
                    </p>
                  </div>
                </div>
              )}

              <div
                style={{
                  marginTop: "auto",
                  display: "flex",
                  gap: "8px",
                  alignItems: "flex-start",
                  color: "#9CA3AF",
                  fontSize: "0.7rem",
                  lineHeight: "1.5",
                }}
              >
                <ShieldCheck
                  size={14}
                  style={{ flexShrink: 0, marginTop: "2px" }}
                />
                <span>
                  Analysis generated by SkinAI Ensemble (v2.1). This is an
                  automated assessment and should not replace professional
                  medical advice.
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
