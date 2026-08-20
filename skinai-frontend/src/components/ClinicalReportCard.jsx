import React, { useRef, useState, useMemo } from 'react';
import { 
  Download, X, ShieldCheck, AlertTriangle, 
  CheckCircle2, User, Calendar, Activity, Stethoscope, 
  Layers, Copy, Check, FileText, Loader2 
} from 'lucide-react';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import { generateReportPdf } from '../api';
import { generateDirectPDFReport } from '../utils/pdfExport';

export default function ClinicalReportCard({ 
  isOpen, 
  onClose, 
  result, 
  imageUrl, 
  patientInfo,
  details,
  finalConfidence,
  isHealthy 
}) {
  const [copied, setCopied] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const reportRef = useRef(null);

  if (!isOpen || !result) return null;

  const reportId = useMemo(() => {
    return 'TR-AI-' + Math.random().toString(36).substring(2, 7).toUpperCase();
  }, []);

  const timestamp = useMemo(() => {
    return new Date().toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }, []);

  const probs = result.probabilities || {};
  const sortedProbs = [
    { code: 'MEL', name: 'Melanoma', pct: (probs.melanoma || probs.mel || 0) * 100 },
    { code: 'BCC', name: 'Basal Cell Carcinoma', pct: (probs.basal_cell_carcinoma || probs.bcc || 0) * 100 },
    { code: 'NV', name: 'Melanocytic Nevi', pct: (probs.nevus || probs.nv || 0) * 100 },
    { code: 'AKIEC', name: 'Actinic Keratoses', pct: (probs.actinic_keratosis || probs.akiec || 0) * 100 },
    { code: 'BKL', name: 'Benign Keratosis', pct: (probs.benign_keratosis || probs.bkl || 0) * 100 },
    { code: 'DF', name: 'Dermatofibroma', pct: (probs.dermatofibroma || probs.df || 0) * 100 },
    { code: 'VASC', name: 'Vascular Lesions', pct: (probs.vascular_lesion || probs.vasc || 0) * 100 },
    { code: 'SCC', name: 'Squamous Cell Carcinoma', pct: (probs.squamous_cell_carcinoma || probs.scc || 0) * 100 },
    { code: 'HEALTHY', name: 'Healthy Skin', pct: (probs.healthy || probs.normal || 0) * 100 },
  ].sort((a, b) => b.pct - a.pct);

  // GradCAM heatmap image
  let gradcamSrc = result.gradcam_image || result.heatmap || result.gradcam || null;
  if (gradcamSrc && typeof gradcamSrc === 'string' && !gradcamSrc.startsWith('data:') && !gradcamSrc.startsWith('http')) {
    gradcamSrc = `data:image/png;base64,${gradcamSrc}`;
  }

  // Instant Vector PDF Download Handler
  const handleDownloadDirectPDF = async () => {
    setDownloadingPdf(true);
    try {
      const ok = generateDirectPDFReport({
        reportId,
        timestamp,
        patientInfo,
        details,
        finalConfidence,
        isHealthy,
        sortedProbs,
        imageUrl,
        gradcamSrc,
      });

      if (!ok) {
        // Fallback to server endpoint
        const blob = await generateReportPdf({
          report_id: reportId,
          timestamp: timestamp,
          patient: patientInfo,
          finding: details.fullName,
          confidence: finalConfidence,
          is_high_risk: details.isMelanoma,
          explanation: details.plainExplanation,
          action: details.patientAction,
          probabilities: sortedProbs,
          image_base64: imageUrl,
          heatmap_base64: gradcamSrc,
        });

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Clinical_Report_${reportId}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('PDF download error:', err);
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleCopySummary = () => {
    const summaryText = `TWACHARAKSHAK AI CLINICAL ASSESSMENT REPORT
Report ID: ${reportId}
Date: ${timestamp}
Patient: ${patientInfo?.name || 'Not provided'} (${patientInfo?.age || '--'} yo, ${patientInfo?.gender || '--'})
Primary Finding: ${details.fullName}
Confidence: ${(finalConfidence * 100).toFixed(1)}%
Risk Level: ${details.isMelanoma ? 'High Risk' : isHealthy ? 'Normal/Healthy' : 'Low Risk / Benign'}
Recommendation: ${details.patientAction}`;

    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div 
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 999,
        background: 'rgba(15, 23, 42, 0.75)',
        backdropFilter: 'blur(12px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '1.5rem',
        overflowY: 'auto',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div 
        style={{
          background: '#ffffff',
          borderRadius: '24px',
          maxWidth: '860px',
          width: '100%',
          maxHeight: '92vh',
          overflowY: 'auto',
          boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.3)',
          border: '1px solid #E5E7EB',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
      >
        {/* ── TOP ACTION TOOLBAR ── */}
        <div 
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '1rem 1.75rem',
            borderBottom: '1px solid #E5E7EB',
            background: '#F9FAFB',
            position: 'sticky',
            top: 0,
            zIndex: 10,
            borderRadius: '24px 24px 0 0',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{
              background: '#EFF6FF',
              color: '#2563EB',
              padding: '4px 10px',
              borderRadius: '999px',
              fontSize: '0.72rem',
              fontWeight: 700,
            }}>
              INTERACTIVE CLINICAL REPORT CARD
            </span>
            <span style={{ fontSize: '0.8125rem', color: '#6B7280', fontWeight: 600 }}>
              ID: {reportId}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              onClick={handleCopySummary}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 16px',
                borderRadius: '999px',
                background: '#fff',
                border: '1px solid #E5E7EB',
                fontSize: '0.8125rem',
                fontWeight: 600,
                color: '#374151',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
            >
              {copied ? <Check size={14} color="#10B981" /> : <Copy size={14} />}
              <span>{copied ? 'Copied' : 'Copy Summary'}</span>
            </button>

            {/* Direct Instant PDF Download Button */}
            <button
              onClick={handleDownloadDirectPDF}
              disabled={downloadingPdf}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '8px 20px',
                borderRadius: '999px',
                background: '#2563EB',
                border: 'none',
                fontSize: '0.8125rem',
                fontWeight: 700,
                color: '#fff',
                cursor: downloadingPdf ? 'wait' : 'pointer',
                boxShadow: '0 4px 14px rgba(37, 99, 235, 0.3)',
                transition: 'all 0.2s',
                opacity: downloadingPdf ? 0.85 : 1,
              }}
            >
              {downloadingPdf ? (
                <>
                  <div className="spinner" style={{ width: '12px', height: '12px' }} />
                  <span>Generating PDF...</span>
                </>
              ) : (
                <>
                  <Download size={14} />
                  <span>Download PDF Report</span>
                </>
              )}
            </button>

            <button
              onClick={onClose}
              style={{
                background: 'transparent',
                border: 'none',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: '#6B7280',
              }}
              title="Close"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        {/* ── CLINICAL REPORT BODY ── */}
        <div 
          ref={reportRef}
          style={{
            padding: '2.5rem 3rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem',
            color: '#111827',
            background: '#ffffff',
          }}
        >
          {/* Header Banner */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            borderBottom: '2px solid #111827',
            paddingBottom: '1.5rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{
                width: '44px',
                height: '44px',
                borderRadius: '12px',
                background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
              </div>
              <div>
                <h2 style={{ fontFamily: 'Syne, sans-serif', fontSize: '1.4rem', fontWeight: 800, margin: 0 }}>
                  Twacha<span style={{ color: '#10B981' }}>Rakshak</span>
                </h2>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#6B7280', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Clinical Dermatological Diagnostic Report
                </span>
              </div>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.8125rem', fontWeight: 700, color: '#111827' }}>
                DOC ID: {reportId}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#6B7280', marginTop: '2px' }}>
                Issued: {timestamp}
              </div>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                background: '#ECFDF5',
                color: '#059669',
                border: '1px solid #A7F3D0',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '0.6875rem',
                fontWeight: 800,
                marginTop: '6px',
              }}>
                <CheckCircle2 size={11} /> AI VALIDATED
              </div>
            </div>
          </div>

          {/* Patient Demographics Box */}
          <div style={{
            background: '#F9FAFB',
            borderRadius: '16px',
            border: '1px solid #E5E7EB',
            padding: '1.25rem 1.5rem',
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '1rem',
          }}>
            <div>
              <span style={{ fontSize: '0.6875rem', fontWeight: 800, color: '#6B7280', textTransform: 'uppercase' }}>PATIENT NAME</span>
              <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#111827', marginTop: '2px' }}>
                {patientInfo?.name || 'Anonymous Patient'}
              </div>
            </div>
            <div>
              <span style={{ fontSize: '0.6875rem', fontWeight: 800, color: '#6B7280', textTransform: 'uppercase' }}>AGE / GENDER</span>
              <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#111827', marginTop: '2px' }}>
                {patientInfo?.age ? `${patientInfo.age} Yrs` : '--'} / {patientInfo?.gender || '--'}
              </div>
            </div>
            <div>
              <span style={{ fontSize: '0.6875rem', fontWeight: 800, color: '#6B7280', textTransform: 'uppercase' }}>MODALITY</span>
              <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#2563EB', marginTop: '2px' }}>
                Dermoscopy (3D-CA)
              </div>
            </div>
            <div>
              <span style={{ fontSize: '0.6875rem', fontWeight: 800, color: '#6B7280', textTransform: 'uppercase' }}>PROTOCOL</span>
              <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#111827', marginTop: '2px' }}>
                ISIC 2019 Ensemble
              </div>
            </div>
          </div>

          {/* Primary Finding Hero Box */}
          <div style={{
            background: isHealthy ? '#ECFDF5' : details.isMelanoma ? '#FEF2F2' : '#EFF6FF',
            border: `1.5px solid ${isHealthy ? '#A7F3D0' : details.isMelanoma ? '#FCA5A5' : '#BFDBFE'}`,
            borderRadius: '18px',
            padding: '1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
          }}>
            <div>
              <span style={{
                fontSize: '0.72rem',
                fontWeight: 800,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                color: isHealthy ? '#065F46' : details.isMelanoma ? '#991B1B' : '#1E40AF',
              }}>
                PRIMARY DIAGNOSTIC FINDING
              </span>
              <h3 style={{
                fontFamily: 'Syne, sans-serif',
                fontSize: '1.75rem',
                fontWeight: 800,
                color: '#111827',
                margin: '4px 0 6px',
              }}>
                {details.fullName}
              </h3>
              <p style={{ fontSize: '0.875rem', color: '#4B5563', margin: 0, maxWidth: '480px' }}>
                {details.plainExplanation}
              </p>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{
                fontSize: '2rem',
                fontWeight: 900,
                fontFamily: 'Syne, sans-serif',
                color: isHealthy ? '#059669' : details.isMelanoma ? '#DC2626' : '#2563EB',
              }}>
                {(finalConfidence * 100).toFixed(1)}%
              </div>
              <span style={{
                display: 'inline-block',
                padding: '4px 12px',
                borderRadius: '999px',
                fontSize: '0.72rem',
                fontWeight: 800,
                textTransform: 'uppercase',
                background: isHealthy ? '#10B981' : details.isMelanoma ? '#DC2626' : '#2563EB',
                color: '#fff',
              }}>
                {details.isMelanoma ? 'High Risk Condition' : isHealthy ? 'Normal / Negative' : 'Low Risk / Benign'}
              </span>
            </div>
          </div>

          {/* Visual Evidence Section (Images Side by Side) */}
          <div>
            <h4 style={{
              fontSize: '0.8125rem',
              fontWeight: 800,
              color: '#6B7280',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              marginBottom: '0.75rem',
            }}>
              Visual Evidence &amp; Spatial Attention
            </h4>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              {/* Original Image */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '14px', overflow: 'hidden', background: '#F8FAFC' }}>
                <div style={{ height: '180px', position: 'relative' }}>
                  <img 
                    src={imageUrl} 
                    alt="Dermoscopy Lesion" 
                    crossOrigin="anonymous"
                    style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} 
                  />
                  <span style={{
                    position: 'absolute',
                    bottom: 8,
                    left: 8,
                    background: 'rgba(0,0,0,0.7)',
                    color: '#fff',
                    fontSize: '0.6875rem',
                    fontWeight: 700,
                    padding: '3px 8px',
                    borderRadius: '4px',
                  }}>
                    Input Dermoscopic Specimen
                  </span>
                </div>
              </div>

              {/* GradCAM Heatmap */}
              <div style={{ border: '1px solid #E5E7EB', borderRadius: '14px', overflow: 'hidden', background: '#000' }}>
                <div style={{ height: '180px', position: 'relative' }}>
                  {gradcamSrc ? (
                    <img 
                      src={gradcamSrc} 
                      alt="Neural Saliency" 
                      crossOrigin="anonymous"
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} 
                    />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#94A3B8', fontSize: '0.75rem' }}>
                      Attention Saliency Map
                    </div>
                  )}
                  <span style={{
                    position: 'absolute',
                    bottom: 8,
                    left: 8,
                    background: 'rgba(0,0,0,0.7)',
                    color: '#fff',
                    fontSize: '0.6875rem',
                    fontWeight: 700,
                    padding: '3px 8px',
                    borderRadius: '4px',
                  }}>
                    3D-CA Neural Saliency Overlay
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Differential Probabilities Breakdown Table */}
          <div>
            <h4 style={{
              fontSize: '0.8125rem',
              fontWeight: 800,
              color: '#6B7280',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
              marginBottom: '0.75rem',
            }}>
              9-Class Differential Diagnostic Probabilities
            </h4>

            <div style={{ border: '1px solid #E5E7EB', borderRadius: '14px', overflow: 'hidden' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8125rem' }}>
                <thead>
                  <tr style={{ background: '#F9FAFB', borderBottom: '1px solid #E5E7EB', textAlign: 'left' }}>
                    <th style={{ padding: '8px 14px', color: '#6B7280', fontWeight: 700 }}>CLASS CODE</th>
                    <th style={{ padding: '8px 14px', color: '#6B7280', fontWeight: 700 }}>PATHOLOGY NAME</th>
                    <th style={{ padding: '8px 14px', color: '#6B7280', fontWeight: 700, width: '40%' }}>PROBABILITY DISTRIBUTION</th>
                    <th style={{ padding: '8px 14px', color: '#6B7280', fontWeight: 700, textAlign: 'right' }}>CONFIDENCE</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedProbs.map((p, idx) => (
                    <tr 
                      key={p.code}
                      style={{
                        borderBottom: idx < sortedProbs.length - 1 ? '1px solid #F3F4F6' : 'none',
                        background: idx === 0 ? 'rgba(37, 99, 235, 0.04)' : 'transparent',
                      }}
                    >
                      <td style={{ padding: '8px 14px', fontWeight: 800, color: idx === 0 ? '#2563EB' : '#6B7280' }}>
                        {p.code}
                      </td>
                      <td style={{ padding: '8px 14px', fontWeight: idx === 0 ? 700 : 500, color: '#111827' }}>
                        {p.name}
                      </td>
                      <td style={{ padding: '8px 14px' }}>
                        <div style={{ height: '6px', background: '#F3F4F6', borderRadius: '3px', overflow: 'hidden' }}>
                          <div style={{
                            height: '100%',
                            width: `${Math.max(1, p.pct)}%`,
                            background: idx === 0 ? '#2563EB' : '#94A3B8',
                            borderRadius: '3px',
                          }} />
                        </div>
                      </td>
                      <td style={{ padding: '8px 14px', textAlign: 'right', fontWeight: 700, color: idx === 0 ? '#2563EB' : '#4B5563' }}>
                        {p.pct.toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Action Plan & Recommendation */}
          <div style={{
            background: '#F9FAFB',
            border: '1px solid #E5E7EB',
            borderRadius: '16px',
            padding: '1.25rem 1.5rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.75rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#111827', fontWeight: 700, fontSize: '0.875rem' }}>
              <Stethoscope size={16} color="#2563EB" />
              <span>Recommended Clinical Next Steps</span>
            </div>
            <p style={{ fontSize: '0.875rem', color: '#374151', margin: 0, lineHeight: 1.6 }}>
              <strong>Patient Care Action:</strong> {details.patientAction}
            </p>
            <p style={{ fontSize: '0.8125rem', color: '#6B7280', margin: 0, lineHeight: 1.5 }}>
              <strong>Key Morphological Signs:</strong> {details.symptoms}
            </p>
          </div>

          {/* Legal / Technical Footer Block */}
          <div style={{
            borderTop: '1px solid #E5E7EB',
            paddingTop: '1.25rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.72rem',
            color: '#9CA3AF',
            flexWrap: 'wrap',
            gap: '1rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <ShieldCheck size={14} color="#6B7280" />
              <span>Research &amp; Clinical Decision-Support Model v2.1 • Not a substitute for histopathological biopsy</span>
            </div>
            <div>
              Generated via TwachaRakshak AI Ensemble Platform
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
