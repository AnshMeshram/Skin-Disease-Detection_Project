import React, { useState } from 'react';
import { 
  BarChart3, PieChart, TrendingUp, Info, Database, Maximize2, 
  Layers, Cpu, Eye, X, Activity, ShieldCheck, CheckCircle2 
} from 'lucide-react';

/* ── Live Dataset Statistics Data ────────────────────────────────────────── */
const SITE_DATA = [
  { label: 'Anterior Torso', count: 6915, color: '#3B82F6' },
  { label: 'Lower Extremity', count: 4990, color: '#10B981' },
  { label: 'Head/Neck', count: 4587, color: '#F59E0B' },
  { label: 'Upper Extremity', count: 2910, color: '#EF4444' },
  { label: 'Posterior Torso', count: 2787, color: '#8B5CF6' },
  { label: 'Other / Acral', count: 511, color: '#6B7280' }
];

const AGE_DIST = [
  { x: 0, y: 54 }, { x: 10, y: 142 }, { x: 20, y: 388 }, 
  { x: 30, y: 1199 }, { x: 40, y: 2246 }, { x: 50, y: 2489 }, 
  { x: 60, y: 2036 }, { x: 70, y: 2120 }, { x: 80, y: 1459 }, { x: 85, y: 1319 }
];

const CLASS_DISTRIBUTION = [
  { name: 'Melanocytic Nevi (NV)', count: 12875, pct: '50.8%', color: '#2563EB' },
  { name: 'Melanoma (MEL)', count: 4522, pct: '17.8%', color: '#DC2626' },
  { name: 'Benign Keratosis (BKL)', count: 2624, pct: '10.3%', color: '#D97706' },
  { name: 'Basal Cell Carcinoma (BCC)', count: 3323, pct: '13.1%', color: '#7C3AED' },
  { name: 'Actinic Keratosis (AKIEC)', count: 867, pct: '3.4%', color: '#059669' },
  { name: 'Vascular Lesions (VASC)', count: 253, pct: '1.0%', color: '#DB2777' },
  { name: 'Dermatofibroma (DF)', count: 239, pct: '0.9%', color: '#475569' },
  { name: 'Squamous Cell Ca. (SCC)', count: 628, pct: '2.5%', color: '#EA580C' }
];

/* ── 6 Core Empirical IEEE Figures for Deep Analysis ─────────────────────── */
const CORE_FIGURES = [
  {
    id: 'dist_bar',
    title: 'ISIC 2019 Class Imbalance',
    category: 'DATASET METRICS',
    src: '/plots/dataset_distribution.png',
    desc: 'Distribution of 25,331 dermatoscopic images across diagnostic categories.'
  },
  {
    id: 'kfold_curves',
    title: '5-Fold Cross Validation Trajectory',
    category: 'MODEL CONVERGENCE',
    src: '/plots/kfold_loss_curves_ieee.png',
    desc: 'Train vs. validation loss trajectory across all 5 cross-validation folds ensuring zero overfitting.'
  },
  {
    id: 'per_class',
    title: 'Per-Class Sensitivity & Specificity',
    category: 'CLINICAL BENCHMARKS',
    src: '/plots/per_class_metrics_ieee.png',
    desc: 'Detailed breakdown of Precision, Recall, and ROC-AUC scores for each diagnostic category.'
  },
  {
    id: 'tsne',
    title: 't-SNE Latent Feature Space',
    category: 'NEURAL EXPLAINABILITY',
    src: '/plots/tsne_visualization_ieee.png',
    desc: '2D manifold projection demonstrating high intra-class clustering and distinct inter-class margins.'
  },
  {
    id: 'preprocessing',
    title: 'Dermoscopic Preprocessing Pipeline',
    category: 'IMAGE PREPROCESSING',
    src: '/plots/preprocessing_pipeline_grid.png',
    desc: 'Step-by-step image processing: DullRazor hair removal, Shades-of-Grey color constancy, and CLAHE.'
  },
  {
    id: 'gradcam',
    title: 'Grad-CAM Neural Attention Saliency',
    category: 'NEURAL EXPLAINABILITY',
    src: '/plots/gradcam_saliency_ieee.png',
    desc: 'Neural spatial activation maps highlighting morphological lesion boundaries.'
  }
];

export default function DatasetAnalysis() {
  const [activeImage, setActiveImage] = useState(null);

  const maxAgeCount = Math.max(...AGE_DIST.map(d => d.y));
  const maxSiteCount = Math.max(...SITE_DATA.map(d => d.count));
  const maxClassCount = Math.max(...CLASS_DISTRIBUTION.map(c => c.count));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '3rem', marginTop: '2rem' }}>
      
      {/* ── SECTION HEADER & METRIC COUNTERS ──────────────────────────────── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: '#2563EB', color: '#fff', padding: '10px', borderRadius: '12px' }}>
            <Database size={24} />
          </div>
          <div>
            <h2 className="syne" style={{ fontSize: '2rem', color: '#111827', margin: 0 }}>
              ISIC 2019 Curation &amp; Analytics
            </h2>
            <p style={{ color: '#6B7280', fontSize: '0.95rem', margin: '4px 0 0' }}>
              Dermatoscopic case distribution, anatomical mapping, and 5-fold cross validation benchmarks
            </p>
          </div>
        </div>

        {/* 4 Summary Stat Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
          {[
            { label: 'TOTAL DERMOSCOPIC SCANS', value: '25,331', sub: 'ISIC 2019 Benchmark' },
            { label: 'DIAGNOSTIC CATEGORIES', value: '9 Classes', sub: '8 Pathologies + Healthy' },
            { label: 'VALIDATION SCHEMA', value: '5-Fold CV', sub: 'Stratified Cross-Val' },
            { label: 'ENSEMBLE AUC SCORE', value: '95.48%', sub: 'Weighted Consensus' }
          ].map((stat, i) => (
            <div key={i} style={{ background: '#fff', padding: '1.25rem', borderRadius: '16px', border: '1px solid #E5E7EB', boxShadow: '0 4px 15px rgba(0,0,0,0.02)' }}>
              <span style={{ fontSize: '0.65rem', fontWeight: 800, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{stat.label}</span>
              <div className="syne" style={{ fontSize: '1.6rem', fontWeight: 800, color: '#2563EB', margin: '4px 0 2px' }}>{stat.value}</div>
              <span style={{ fontSize: '0.72rem', color: '#94A3B8', fontWeight: 600 }}>{stat.sub}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── INTERACTIVE LIVE CHARTS GRID ───────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '2rem' }}>
        
        {/* Donut Chart - Regional Distribution */}
        <div style={{ background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.75rem' }}>
            <PieChart size={20} color="#2563EB" />
            <h3 className="syne" style={{ fontSize: '1.2rem', margin: 0 }}>Anatomical Site Proportions</h3>
          </div>
          
          <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center' }}>
            <div style={{ position: 'relative', width: '150px', height: '150px' }}>
              <svg viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                {(() => {
                  let offset = 0;
                  const total = SITE_DATA.reduce((acc, s) => acc + s.count, 0);
                  return SITE_DATA.map((s, i) => {
                    const percent = (s.count / total) * 100;
                    const dashoffset = -offset;
                    offset += percent;
                    return (
                      <circle 
                        key={i} cx="18" cy="18" r="15.9" 
                        fill="transparent" stroke={s.color} strokeWidth="4" 
                        strokeDasharray={`${percent} 100`} strokeDashoffset={dashoffset} 
                      />
                    );
                  });
                })()}
              </svg>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1 }}>
              {SITE_DATA.map((s, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: s.color }} />
                  <span style={{ fontSize: '0.75rem', color: '#4B5563', flex: 1 }}>{s.label}</span>
                  <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#111827' }}>
                    {((s.count / SITE_DATA.reduce((a,b)=>a+b.count,0))*100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bar Chart - Occurrence Intensity */}
        <div style={{ background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.75rem' }}>
            <BarChart3 size={20} color="#2563EB" />
            <h3 className="syne" style={{ fontSize: '1.2rem', margin: 0 }}>Anatomical Occurrence Intensity</h3>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {SITE_DATA.map((s, i) => (
              <div key={i}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: '4px' }}>
                  <span style={{ fontWeight: 600, color: '#4B5563' }}>{s.label}</span>
                  <span style={{ fontWeight: 700, color: s.color }}>{s.count.toLocaleString()} Cases</span>
                </div>
                <div style={{ height: '6px', background: '#F3F4F6', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(s.count / maxSiteCount) * 100}%`, background: s.color, borderRadius: '3px' }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Line Chart - Age-wise Case Prevalence */}
        <div style={{ gridColumn: '1 / -1', background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.75rem' }}>
            <TrendingUp size={20} color="#2563EB" />
            <h3 className="syne" style={{ fontSize: '1.2rem', margin: 0 }}>Age-wise Case Prevalence Curve</h3>
          </div>
          <div style={{ height: '200px', width: '100%', display: 'flex', alignItems: 'flex-end', gap: '10px', position: 'relative' }}>
            <svg width="100%" height="100%" viewBox="0 0 1000 200" preserveAspectRatio="none">
              <path 
                d={`M ${AGE_DIST.map((d, i) => `${(i / (AGE_DIST.length - 1)) * 1000},${200 - (d.y / maxAgeCount) * 180}`).join(' L ')}`}
                fill="none" stroke="#2563EB" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round"
              />
              {AGE_DIST.map((d, i) => (
                <circle 
                  key={i} cx={(i / (AGE_DIST.length - 1)) * 1000} cy={200 - (d.y / maxAgeCount) * 180} 
                  r="6" fill="#fff" stroke="#2563EB" strokeWidth="3"
                />
              ))}
            </svg>
            <div style={{ position: 'absolute', bottom: '-25px', left: 0, right: 0, display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#9CA3AF' }}>
               {AGE_DIST.map((d, i) => <span key={i}>{d.x}y</span>)}
            </div>
          </div>
        </div>

        {/* Class Imbalance Distribution Bars */}
        <div style={{ gridColumn: '1 / -1', background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.75rem' }}>
            <Layers size={20} color="#2563EB" />
            <h3 className="syne" style={{ fontSize: '1.2rem', margin: 0 }}>Diagnostic Class Imbalance Breakdown</h3>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.25rem' }}>
            {CLASS_DISTRIBUTION.map((c, i) => (
              <div key={i} style={{ background: '#F8FAFC', padding: '1rem 1.25rem', borderRadius: '14px', border: '1px solid #E2E8F0' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 700, color: '#1E293B' }}>{c.name}</span>
                  <span style={{ fontWeight: 800, color: c.color }}>{c.pct}</span>
                </div>
                <div style={{ height: '6px', background: '#E2E8F0', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${(c.count / maxClassCount) * 100}%`, background: c.color, borderRadius: '3px' }} />
                </div>
                <span style={{ fontSize: '0.68rem', color: '#64748B', display: 'inline-block', marginTop: '4px' }}>
                  {c.count.toLocaleString()} Verified Samples
                </span>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* ── 6 CORE EMPIRICAL IEEE FIGURE GALLERY ──────────────────────────── */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.5rem' }}>
          <Activity size={22} color="#2563EB" />
          <h3 className="syne" style={{ fontSize: '1.5rem', color: '#0F172A', margin: 0 }}>
            Empirical Validation Figures
          </h3>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(310px, 1fr))',
          gap: '1.75rem'
        }}>
          {CORE_FIGURES.map((fig) => (
            <div
              key={fig.id}
              style={{
                background: '#fff',
                borderRadius: '24px',
                border: '1px solid #E5E7EB',
                boxShadow: '0 8px 30px rgba(0,0,0,0.03)',
                overflow: 'hidden',
                display: 'flex',
                flexDirection: 'column',
                cursor: 'pointer',
                transition: 'transform 0.3s ease, boxShadow 0.3s ease'
              }}
              onClick={() => setActiveImage(fig)}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-4px)';
                e.currentTarget.style.boxShadow = '0 16px 40px rgba(37,99,235,0.12)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 8px 30px rgba(0,0,0,0.03)';
              }}
            >
              <div style={{ position: 'relative', height: '210px', background: '#F8FAFC', borderBottom: '1px solid #F1F5F9', overflow: 'hidden' }}>
                <img
                  src={fig.src}
                  alt={fig.title}
                  style={{ width: '100%', height: '100%', objectFit: 'contain', padding: '10px', display: 'block' }}
                />
                <div style={{ position: 'absolute', top: 12, left: 12 }}>
                  <span style={{
                    background: 'rgba(15, 23, 42, 0.75)',
                    backdropFilter: 'blur(8px)',
                    color: '#fff',
                    fontSize: '0.62rem',
                    fontWeight: 800,
                    padding: '4px 10px',
                    borderRadius: '6px',
                    letterSpacing: '0.05em'
                  }}>
                    {fig.category}
                  </span>
                </div>
                <div style={{
                  position: 'absolute', bottom: 12, right: 12,
                  background: '#fff', color: '#2563EB',
                  borderRadius: '50%', width: 32, height: 32,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                }}>
                  <Maximize2 size={15} />
                </div>
              </div>

              <div style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.35rem', flex: 1 }}>
                <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0 }}>
                  {fig.title}
                </h4>
                <p style={{ fontSize: '0.8rem', color: '#64748B', lineHeight: 1.5, margin: 0 }}>
                  {fig.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── REGIONAL PATHOLOGY & DEMOGRAPHIC INSIGHTS ───────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
        <div style={{ background: '#F8FAFC', padding: '2rem', borderRadius: '24px', border: '1px solid #E2E8F0' }}>
          <h4 className="syne" style={{ fontSize: '1.1rem', color: '#1E293B', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Info size={18} color="#3B82F6" /> Regional Pathologies
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              { site: 'Head/Neck', dominant: 'BCC (Basal Cell Carcinoma)', desc: 'High correlation with chronic solar ultraviolet exposure' },
              { site: 'Palms/Soles', dominant: 'MEL (Melanoma)', desc: 'Acral Lentiginous subtype prevalence' },
              { site: 'Torso/Extremities', dominant: 'NV (Common Nevus)', desc: 'Standard benign skin growth patterns' }
            ].map((item, i) => (
              <div key={i} style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 800, color: '#64748B', textTransform: 'uppercase' }}>{item.site}</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 800, color: '#1E293B', margin: '4px 0' }}>{item.dominant}</div>
                <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{ background: '#F8FAFC', padding: '2rem', borderRadius: '24px', border: '1px solid #E2E8F0' }}>
          <h4 className="syne" style={{ fontSize: '1.1rem', color: '#1E293B', marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingUp size={18} color="#10B981" /> Demographic Vulnerability Peaks
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {[
              { age: 'Age Group 50–70 Years', condition: 'Actinic Keratosis & BCC Peak', note: 'Cumulative sun damage onset' },
              { age: 'Age Group 20–40 Years', condition: 'Melanocytic Nevi Peak', note: 'High mole manifestation period' },
              { age: 'Age Group 65+ Years', condition: 'Invasive Melanoma Peak', note: 'Higher risk requiring annual screening' }
            ].map((item, i) => (
              <div key={i} style={{ background: '#fff', padding: '1rem', borderRadius: '12px', border: '1px solid #E2E8F0' }}>
                <div style={{ fontSize: '0.72rem', fontWeight: 800, color: '#10B981', textTransform: 'uppercase' }}>{item.age}</div>
                <div style={{ fontSize: '0.9rem', fontWeight: 800, color: '#1E293B', margin: '4px 0' }}>{item.condition}</div>
                <div style={{ fontSize: '0.72rem', color: '#94A3B8' }}>{item.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Fullscreen Lightbox Modal */}
      {activeImage && (
        <div
          onClick={() => setActiveImage(null)}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(15, 23, 42, 0.85)',
            backdropFilter: 'blur(12px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem'
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#fff',
              borderRadius: '24px',
              maxWidth: '1000px',
              width: '100%',
              maxHeight: '90vh',
              overflow: 'auto',
              padding: '2rem',
              position: 'relative',
              boxShadow: '0 25px 50px rgba(0,0,0,0.3)',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <span style={{ fontSize: '0.72rem', fontWeight: 800, color: '#2563EB', textTransform: 'uppercase' }}>
                  {activeImage.category}
                </span>
                <h3 className="syne" style={{ fontSize: '1.5rem', color: '#0F172A', margin: 0 }}>
                  {activeImage.title}
                </h3>
              </div>
              <button
                onClick={() => setActiveImage(null)}
                style={{
                  background: '#F1F5F9',
                  border: 'none',
                  borderRadius: '50%',
                  width: 36,
                  height: 36,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  color: '#475569'
                }}
              >
                <X size={20} />
              </button>
            </div>

            <div style={{ background: '#F8FAFC', borderRadius: '16px', padding: '1rem', border: '1px solid #E2E8F0', display: 'flex', justifyContent: 'center' }}>
              <img
                src={activeImage.src}
                alt={activeImage.title}
                style={{ maxWidth: '100%', maxHeight: '65vh', objectFit: 'contain', borderRadius: '8px' }}
              />
            </div>

            <p style={{ fontSize: '0.92rem', color: '#475569', lineHeight: 1.6, margin: 0 }}>
              {activeImage.desc}
            </p>
          </div>
        </div>
      )}

    </div>
  );
}
