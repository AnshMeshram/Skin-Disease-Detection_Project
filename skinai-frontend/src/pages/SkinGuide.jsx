import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useReveal } from '../hooks/useReveal';
import { Info, AlertCircle, CheckCircle2, AlertTriangle, ArrowUpRight, Eye, Shield, ChevronRight, LayoutGrid, Rows3 } from 'lucide-react';

/* ── Disease Image Paths (public/diseases/) ──────── */
const DISEASE_IMAGES = {
  'Melanoma': '/diseases/melanoma.png',
  'Melanocytic Nevus': '/diseases/nevus.png',
  'Basal Cell Carcinoma': '/diseases/bcc.png',
  'Actinic Keratosis': '/diseases/actinic_keratosis.png',
  'Benign Keratosis': '/diseases/benign_keratosis.png',
  'Dermatofibroma': '/diseases/dermatofibroma.png',
  'Vascular Lesion': '/diseases/vascular_lesion.png',
  'Squamous Cell Carcinoma': '/diseases/scc.png',
  'Healthy Skin': '/diseases/healthy_skin.png',
};

/* ── Enhanced Disease Data ───────────────────────── */
const DISEASES = [
  {
    name: 'Melanoma',
    short: 'The most serious type of skin cancer.',
    details: 'The most serious type of skin cancer, develops in the cells (melanocytes) that produce melanin. Often presents as a new, unusual growth or a rapid change in an existing mole. Immediate clinical evaluation required.',
    risk: 'Critical',
    riskLevel: 'high-risk',
    category: 'MALIGNANT NEOPLASM',
    symptoms: ['Asymmetrical shape', 'Irregular borders', 'Color variation', 'Diameter > 6mm'],
    clinicalObs: [
      'Asymmetrical shape with variable pigmentation',
      'Irregular, notched, or scalloped borders',
      'Evolution in size, shape, or color over time'
    ],
    tags: ['Asymmetric', 'Multi-color', 'Evolving'],
    color: '#EF4444',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #2d1b1b 40%, #3d1515 100%)'
  },
  {
    name: 'Melanocytic Nevus',
    short: 'Commonly known as a mole.',
    details: 'A benign growth on the skin formed by a cluster of melanocytes. Generally harmless, representing a normal cluster of melanocytes. Monitor for ABCDE changes.',
    risk: 'Benign',
    riskLevel: 'benign',
    category: 'BENIGN MELANOCYTIC LESION',
    symptoms: ['Uniform tan or brown color', 'Round or oval shape', 'Distinct borders'],
    clinicalObs: [
      'Uniform pigmentation with regular borders',
      'Stable size and shape over time',
      'Typically symmetric and well-circumscribed'
    ],
    tags: ['Symmetrical', 'Even Color', 'Stable'],
    color: '#10B981',
    gradient: 'linear-gradient(145deg, #0f2027 0%, #0a1f14 40%, #1a3a2a 100%)'
  },
  {
    name: 'Basal Cell Carcinoma',
    short: 'Common, slow-growing skin cancer.',
    details: 'Often appears as a slightly transparent bump on the skin, primarily on sun-exposed areas. Originates in basal cells that produce new skin cells.',
    risk: 'High',
    riskLevel: 'high-risk',
    category: 'MALIGNANT NEOPLASM',
    symptoms: ['Pearly or waxy bump', 'Flesh-colored lesion', 'Bleeding/scabby sore'],
    clinicalObs: [
      'Pearly, translucent papule or nodule',
      'Rolled borders with telangiectasia',
      'Central ulceration or crusting possible'
    ],
    tags: ['Pearly Surface', 'Sun Exposed', 'Slow Growing'],
    color: '#F59E0B',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #2e2410 40%, #3d3015 100%)'
  },
  {
    name: 'Actinic Keratosis',
    short: 'Precancerous scaly skin patch.',
    details: 'Caused by years of sun exposure. Develops in sun-damaged skin and can progress to Squamous Cell Carcinoma if left untreated. Early treatment recommended.',
    risk: 'Moderate',
    riskLevel: 'moderate',
    category: 'PRE-MALIGNANT LESION',
    symptoms: ['Rough, dry patch', 'Itching or burning', 'Hard, wart-like surface'],
    clinicalObs: [
      'Rough, scaly patch on sun-damaged skin',
      'Erythematous base with adherent scale',
      'May be easier to feel than see initially'
    ],
    tags: ['UV Linked', 'Scaly', 'Pre-cancerous'],
    color: '#3B82F6',
    gradient: 'linear-gradient(145deg, #0f1a2e 0%, #0f1f3d 40%, #152a4a 100%)'
  },
  {
    name: 'Benign Keratosis',
    short: 'Non-cancerous skin growth.',
    details: 'Waxy, scaly, slightly elevated growths often appearing in older adults. Also known as Seborrheic Keratosis. Completely harmless despite sometimes alarming appearance.',
    risk: 'Benign',
    riskLevel: 'benign',
    category: 'BENIGN EPIDERMAL LESION',
    symptoms: ['Waxy appearance', 'Brown, black or tan color', 'Stuck-on appearance'],
    clinicalObs: [
      'Well-demarcated, "stuck-on" appearance',
      'Waxy or verrucous surface texture',
      'Horn pseudocysts visible on dermoscopy'
    ],
    tags: ['Waxy', 'Stuck-on', 'Harmless'],
    color: '#6366F1',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #1e1a3d 40%, #2a1f4a 100%)'
  },
  {
    name: 'Dermatofibroma',
    short: 'Common benign fibrous nodule.',
    details: 'A benign skin growth often found on the legs. Usually develops after a minor injury like an insect bite. Firm to touch with a characteristic dimple sign.',
    risk: 'Benign',
    riskLevel: 'benign',
    category: 'BENIGN FIBROUS LESION',
    symptoms: ['Firm, small bump', 'Dimples when pinched', 'Variable color (pink to brown)'],
    clinicalObs: [
      'Firm dermal nodule with central white patch',
      'Positive dimple sign on lateral compression',
      'Peripheral pigment network on dermoscopy'
    ],
    tags: ['Firm', 'Dimple Sign', 'Post-trauma'],
    color: '#8B5CF6',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #251a3d 40%, #30204a 100%)'
  },
  {
    name: 'Vascular Lesion',
    short: 'Abnormal growth of blood vessels.',
    details: 'Includes cherry angiomas, hemangiomas, and port-wine stains. Usually harmless vascular proliferations that rarely require treatment unless for cosmetic reasons.',
    risk: 'Low',
    riskLevel: 'low',
    category: 'VASCULAR PROLIFERATION',
    symptoms: ['Red, purple or blue color', 'Blanches on pressure', 'Soft or raised texture'],
    clinicalObs: [
      'Red to purple coloration from blood vessels',
      'Blanching on diascopy (glass pressure)',
      'Lacunae structures visible on dermoscopy'
    ],
    tags: ['Vascular', 'Blanching', 'Soft'],
    color: '#EC4899',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #2e1028 40%, #3d1535 100%)'
  },
  {
    name: 'Squamous Cell Carcinoma',
    short: 'Second most common skin cancer.',
    details: 'Develops in squamous cells making up the middle and outer skin layers. Can be aggressive if left untreated. Strongly associated with chronic UV exposure.',
    risk: 'High',
    riskLevel: 'high-risk',
    category: 'MALIGNANT NEOPLASM',
    symptoms: ['Firm, red nodule', 'Flat sore with scaly crust', 'Non-healing ulcer'],
    clinicalObs: [
      'Keratinizing or non-keratinizing tumor',
      'Irregular vascular pattern on dermoscopy',
      'White structureless areas with ulceration'
    ],
    tags: ['Keratinizing', 'UV Linked', 'Aggressive'],
    color: '#B91C1C',
    gradient: 'linear-gradient(145deg, #1a1a2e 0%, #2e1010 40%, #401515 100%)'
  },
  {
    name: 'Healthy Skin',
    short: 'Normal uninfected skin surface.',
    details: 'Skin with normal texture, pigmentation, and no signs of lesions or inflammation. Represents the baseline for comparison in dermatoscopic evaluation.',
    risk: 'None',
    riskLevel: 'none',
    category: 'NORMAL TISSUE',
    symptoms: ['Smooth texture', 'Uniform pigmentation', 'No inflammation'],
    clinicalObs: [
      'Regular dermatoglyphics and pigment pattern',
      'No atypical structures or vascular changes',
      'Uniform furrow pattern on dermoscopy'
    ],
    tags: ['Healthy', 'Uniform', 'Normal'],
    color: '#059669',
    gradient: 'linear-gradient(145deg, #0f2027 0%, #0a2015 40%, #153020 100%)'
  }
];

/* ── Risk Badge Config ───────────────────────────── */
const RISK_CFG = {
  'high-risk': { bg: '#EF4444', label: 'High Risk' },
  'moderate':  { bg: '#F59E0B', label: 'Moderate' },
  'benign':    { bg: '#10B981', label: 'Benign' },
  'low':       { bg: '#6366F1', label: 'Low Risk' },
  'none':      { bg: '#059669', label: 'No Risk' },
};

/* ── Specimen Image Card Component ────────────────── */
function SpecimenCard({ disease, index, height = 280, style = {} }) {
  const rc = RISK_CFG[disease.riskLevel] || RISK_CFG['low'];
  const imgSrc = DISEASE_IMAGES[disease.name];
  return (
    <div style={{
      background: disease.gradient,
      height,
      position: 'relative',
      overflow: 'hidden',
      ...style,
    }}>
      {/* Clinical Image */}
      <img
        src={imgSrc}
        alt={`${disease.name} clinical reference`}
        style={{
          width: '100%', height: '100%',
          objectFit: 'cover', objectPosition: 'center',
          display: 'block',
        }}
      />
      {/* Subtle gradient overlay for text readability */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'linear-gradient(180deg, rgba(0,0,0,0.15) 0%, transparent 40%, transparent 60%, rgba(0,0,0,0.3) 100%)',
      }} />
      {/* Large watermark number */}
      <span className="syne" style={{
        position: 'absolute', bottom: '8px', right: '16px',
        fontSize: '5.5rem', fontWeight: 800, lineHeight: 1,
        color: 'rgba(255,255,255,0.08)',
        userSelect: 'none',
      }}>
        {String(index + 1).padStart(2, '0')}
      </span>
      {/* Risk badge */}
      <span style={{
        position: 'absolute', top: '14px', left: '14px',
        fontSize: '0.68rem', fontWeight: 700, color: '#fff',
        background: rc.bg, padding: '5px 12px', borderRadius: '8px',
        display: 'flex', alignItems: 'center', gap: '5px',
        boxShadow: `0 2px 8px ${rc.bg}50`,
        zIndex: 1,
      }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#fff', opacity: 0.9 }} />
        {rc.label}
      </span>
      {/* Bottom label */}
      <span style={{
        position: 'absolute', bottom: '14px', left: '14px',
        fontSize: '0.7rem', fontWeight: 600, zIndex: 1,
        color: 'rgba(255,255,255,0.85)',
        background: 'rgba(0,0,0,0.35)',
        backdropFilter: 'blur(10px)',
        padding: '5px 14px', borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.1)',
        letterSpacing: '0.3px',
      }}>
        CASE #{String(index + 1).padStart(2, '0')}
      </span>
    </div>
  );
}

/* ── Pair Card Component ─────────────────────────── */
function PairCard({ disease, index, stagger, onSelect }) {
  return (
    <div className="case-card" style={{ marginTop: stagger ? '2.5rem' : 0 }}>
      <SpecimenCard disease={disease} index={index} height={220} />
      <div style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
          <span style={{
            fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.8px',
            color: '#9CA3AF', textTransform: 'uppercase',
          }}>
            {disease.category}
          </span>
        </div>
        <h3 className="syne" style={{ fontSize: '1.15rem', color: '#111827', marginBottom: '0.5rem' }}>
          {disease.name}
        </h3>
        <p style={{ fontSize: '0.82rem', color: '#6B7280', lineHeight: 1.6, marginBottom: '1rem' }}>
          {disease.short} {disease.details.split('.').slice(1, 2).join('.')}.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '1rem' }}>
          {disease.tags.map((t, i) => (
            <span key={i} className="clinical-tag">
              <CheckCircle2 size={12} color={disease.color} /> {t}
            </span>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button className="case-learn-btn" style={{ color: disease.color }} onClick={() => onSelect(disease)}>
            Learn More <ChevronRight size={14} />
          </button>
          <button className="case-learn-btn" style={{ color: '#9CA3AF' }} onClick={() => onSelect(disease)}>
            Case Study <ArrowUpRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}

/* ── Wide Card Component ─────────────────────────── */
function WideCard({ disease, index, reversed, onSelect }) {
  const imageBlock = (
    <SpecimenCard disease={disease} index={index} height={260} style={{ minHeight: '100%' }} />
  );
  const textBlock = (
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', justifyContent: 'center', position: 'relative' }}>
      {/* Watermark number */}
      <span className="syne" style={{
        position: 'absolute', top: '10px', right: '20px',
        fontSize: '4.5rem', fontWeight: 800, lineHeight: 1,
        color: 'rgba(0,0,0,0.04)', userSelect: 'none',
      }}>
        {String(index + 1).padStart(2, '0')}
      </span>
      <span style={{
        fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.8px',
        color: '#9CA3AF', textTransform: 'uppercase', marginBottom: '6px',
      }}>
        {disease.category}
      </span>
      <h3 className="syne" style={{ fontSize: '1.3rem', color: '#111827', marginBottom: '0.5rem' }}>
        {disease.name}
      </h3>
      <p style={{ fontSize: '0.85rem', color: '#6B7280', lineHeight: 1.65, marginBottom: '1.25rem' }}>
        {disease.details}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '1rem' }}>
        {disease.tags.map((t, i) => (
          <span key={i} className="clinical-tag">
            <CheckCircle2 size={12} color={disease.color} /> {t}
          </span>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
        <button className="case-learn-btn" style={{ color: disease.color }} onClick={() => onSelect(disease)}>
          Learn More <ChevronRight size={14} />
        </button>
        <button className="case-learn-btn" style={{ color: '#9CA3AF' }} onClick={() => onSelect(disease)}>
          Case Study <ArrowUpRight size={14} />
        </button>
      </div>
    </div>
  );

  return (
    <div className="case-wide" style={reversed ? { direction: 'rtl' } : {}}>
      <div style={reversed ? { direction: 'ltr' } : {}}>{imageBlock}</div>
      <div style={reversed ? { direction: 'ltr' } : {}}>{textBlock}</div>
    </div>
  );
}

/* ══════════════════════════════════════════════════ */
/*  MAIN COMPONENT                                   */
/* ══════════════════════════════════════════════════ */
export default function SkinGuide() {
  const navigate = useNavigate();
  const ref = useReveal();
  const [selectedDisease, setSelectedDisease] = React.useState(null);
  const [viewMode, setViewMode] = React.useState('case'); // 'grid' or 'case'

  React.useEffect(() => {
    const hash = window.location.hash;
    if (hash) {
      const id = hash.replace('#', '');
      const element = document.getElementById(id);
      if (element) {
        setTimeout(() => { element.scrollIntoView({ behavior: 'smooth' }); }, 100);
      }
    }
  }, []);

  const hero = DISEASES[0]; // Melanoma

  return (
    <div className="mesh-bg" style={{ minHeight: '100vh', padding: '100px 2rem 4rem' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>

        {/* ══════════════════════════════════════════ */}
        {/* ── PAGE HEADER ────────────────────────── */}
        {/* ══════════════════════════════════════════ */}
        <div style={{ marginBottom: '3rem', position: 'relative' }}>
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: '6px',
            background: '#ECFDF5', color: '#059669',
            padding: '6px 14px', borderRadius: '999px',
            fontSize: '0.72rem', fontWeight: 700,
            marginBottom: '1rem', letterSpacing: '0.5px',
          }}>
            🔬 THE CLINICAL LENS
          </span>

          <h1 className="syne" style={{
            fontSize: 'clamp(2rem, 5vw, 3.2rem)',
            color: '#111827', lineHeight: 1.15,
            marginBottom: '1rem', maxWidth: '600px',
          }}>
            Case Files &<br/>Observation Sheets
          </h1>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '1rem' }}>
            <p style={{
              color: '#6B7280', fontSize: '0.92rem', lineHeight: 1.6,
              maxWidth: '550px',
            }}>
              Interactive clinical reference for high-fidelity dermatological identification.
              Analyse textures, borders, and risk indicators across common morphologies.
            </p>
            {/* View mode toggle */}
            <div style={{ display: 'flex', gap: '4px' }}>
              <button
                onClick={() => setViewMode('grid')}
                style={{
                  width: 36, height: 36, borderRadius: '10px',
                  border: '1px solid #E5E7EB',
                  background: viewMode === 'grid' ? '#111827' : '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', transition: 'all 0.2s ease',
                }}
                title="Grid View"
              >
                <LayoutGrid size={16} color={viewMode === 'grid' ? '#fff' : '#374151'} />
              </button>
              <button
                onClick={() => setViewMode('case')}
                style={{
                  width: 36, height: 36, borderRadius: '10px',
                  border: '1px solid #E5E7EB',
                  background: viewMode === 'case' ? '#111827' : '#fff',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', transition: 'all 0.2s ease',
                }}
                title="Case Files View"
              >
                <Rows3 size={16} color={viewMode === 'case' ? '#fff' : '#374151'} />
              </button>
            </div>
          </div>
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── GRID VIEW (Original Card Layout) ────── */}
        {/* ══════════════════════════════════════════ */}
        {viewMode === 'grid' && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '2.5rem' }}>
              {DISEASES.map((d, i) => (
                <div key={i} id={`grid-${d.name.toLowerCase().replace(/\s+/g, '-')}`} className="case-card" style={{
                  display: 'flex', flexDirection: 'column', scrollMarginTop: '120px',
                }}>
                  {/* Image */}
                  <div style={{ height: 180, position: 'relative', overflow: 'hidden' }}>
                    <img
                      src={DISEASE_IMAGES[d.name]}
                      alt={`${d.name} clinical reference`}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                    />
                    <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(180deg, transparent 50%, rgba(0,0,0,0.2) 100%)' }} />
                    <span style={{
                      position: 'absolute', top: 10, left: 10,
                      fontSize: '0.65rem', fontWeight: 700, color: '#fff',
                      background: (RISK_CFG[d.riskLevel] || RISK_CFG['low']).bg,
                      padding: '4px 10px', borderRadius: '6px',
                      display: 'flex', alignItems: 'center', gap: '4px',
                    }}>
                      <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#fff' }} />
                      {(RISK_CFG[d.riskLevel] || RISK_CFG['low']).label}
                    </span>
                  </div>

                  <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h2 className="syne" style={{ fontSize: '1.15rem', color: '#111827' }}>{d.name}</h2>
                      <span style={{
                        background: `${d.color}15`, color: d.color,
                        fontSize: '0.65rem', fontWeight: 700,
                        padding: '3px 10px', borderRadius: '999px',
                      }}>
                        Risk: {d.risk}
                      </span>
                    </div>

                    <div>
                      <p style={{ fontWeight: 600, fontSize: '0.88rem', color: '#374151', marginBottom: '0.4rem' }}>{d.short}</p>
                      <p style={{ fontSize: '0.82rem', color: '#6B7280', lineHeight: 1.6 }}>{d.details}</p>
                    </div>

                    <div style={{ background: '#F9FAFB', borderRadius: '12px', padding: '1rem' }}>
                      <p style={{ fontSize: '0.72rem', fontWeight: 700, color: '#9CA3AF', textTransform: 'uppercase', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <AlertCircle size={13} /> Key Symptoms
                      </p>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {d.symptoms.map((s, idx) => (
                          <li key={idx} style={{ fontSize: '0.78rem', color: '#4B5563', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <CheckCircle2 size={13} color={d.color} />
                            {s}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <button
                      onClick={() => setSelectedDisease(d)}
                      style={{
                        marginTop: 'auto',
                        background: 'transparent',
                        border: `1px solid ${d.color}`,
                        color: d.color,
                        borderRadius: '999px',
                        padding: '0.55rem',
                        fontSize: '0.78rem',
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.2s',
                      }}
                      onMouseEnter={e => { e.currentTarget.style.background = d.color; e.currentTarget.style.color = '#fff'; }}
                      onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = d.color; }}
                    >
                      Learn More
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* ══════════════════════════════════════════ */}
        {/* ── CASE FILES VIEW ─────────────────────── */}
        {/* ══════════════════════════════════════════ */}
        {viewMode === 'case' && (
          <>

        {/* ── HERO CARD (MELANOMA) ───────────────── */}
        <div id="melanoma" className="case-hero-card" style={{ marginBottom: '2.5rem', scrollMarginTop: '120px' }}>
          <SpecimenCard disease={hero} index={0} height={420} style={{ minHeight: '100%' }} />
          <div style={{ padding: '2rem 2.5rem', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '0.75rem' }}>
              <span style={{
                fontSize: '0.65rem', fontWeight: 700, letterSpacing: '1px',
                color: '#9CA3AF', textTransform: 'uppercase',
              }}>
                {hero.category}
              </span>
              <div style={{
                width: 28, height: 28, borderRadius: '50%',
                border: '1px solid #E5E7EB', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
              }}>
                <ArrowUpRight size={14} color="#9CA3AF" />
              </div>
            </div>
            <h2 className="syne" style={{ fontSize: '1.75rem', color: '#111827', marginBottom: '0.75rem' }}>
              {hero.name}
            </h2>
            <p style={{ fontSize: '0.88rem', color: '#6B7280', lineHeight: 1.65, marginBottom: '1.5rem' }}>
              {hero.details}
            </p>
            <div style={{
              background: '#F8FAFC', borderRadius: '16px', padding: '1.25rem',
              border: '1px solid #F1F5F9',
            }}>
              <h4 style={{
                fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.8px',
                color: '#64748B', textTransform: 'uppercase',
                marginBottom: '0.85rem', display: 'flex', alignItems: 'center', gap: '6px',
              }}>
                ✦ CLINICAL OBSERVATIONS
              </h4>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {hero.clinicalObs.map((obs, i) => (
                  <li key={i} style={{ fontSize: '0.84rem', color: '#374151', display: 'flex', alignItems: 'flex-start', gap: '8px', lineHeight: 1.5 }}>
                    <span style={{ color: hero.color, fontWeight: 800, fontSize: '0.7rem', marginTop: '3px' }}>●</span>
                    {obs}
                  </li>
                ))}
              </ul>
            </div>
            <button
              className="case-learn-btn"
              style={{ color: hero.color, marginTop: '1.25rem', fontSize: '0.85rem' }}
              onClick={() => setSelectedDisease(hero)}
            >
              Full Case Study <ChevronRight size={16} />
            </button>
          </div>
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── PAIR 1: Melanocytic Nevus + BCC ────── */}
        {/* ══════════════════════════════════════════ */}
        <div className="case-pair-grid" style={{ marginBottom: '2.5rem' }}>
          <div id="melanocytic-nevus" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[1]} index={1} stagger={false} onSelect={setSelectedDisease} />
          </div>
          <div id="basal-cell-carcinoma" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[2]} index={2} stagger={true} onSelect={setSelectedDisease} />
          </div>
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── WIDE: Actinic Keratosis ────────────── */}
        {/* ══════════════════════════════════════════ */}
        <div id="actinic-keratosis" style={{ marginBottom: '2.5rem', scrollMarginTop: '120px' }}>
          <WideCard disease={DISEASES[3]} index={3} reversed={false} onSelect={setSelectedDisease} />
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── PAIR 2: Benign Keratosis + Dermato ─── */}
        {/* ══════════════════════════════════════════ */}
        <div className="case-pair-grid" style={{ marginBottom: '2.5rem' }}>
          <div id="benign-keratosis" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[4]} index={4} stagger={false} onSelect={setSelectedDisease} />
          </div>
          <div id="dermatofibroma" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[5]} index={5} stagger={true} onSelect={setSelectedDisease} />
          </div>
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── WIDE: Vascular Lesion (reversed) ───── */}
        {/* ══════════════════════════════════════════ */}
        <div id="vascular-lesion" style={{ marginBottom: '2.5rem', scrollMarginTop: '120px' }}>
          <WideCard disease={DISEASES[6]} index={6} reversed={true} onSelect={setSelectedDisease} />
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── PAIR 3: SCC + Healthy Skin ─────────── */}
        {/* ══════════════════════════════════════════ */}
        <div className="case-pair-grid" style={{ marginBottom: '3.5rem' }}>
          <div id="squamous-cell-carcinoma" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[7]} index={7} stagger={false} onSelect={setSelectedDisease} />
          </div>
          <div id="healthy-skin" style={{ scrollMarginTop: '120px' }}>
            <PairCard disease={DISEASES[8]} index={8} stagger={true} onSelect={setSelectedDisease} />
          </div>
        </div>

        {/* ══════════════════════════════════════════ */}
        {/* ── CLINICAL COMPARISON VIEW ────────────── */}
        {/* ══════════════════════════════════════════ */}
        <div style={{
          background: '#fff', borderRadius: '24px', padding: '2.5rem',
          border: '1px solid #E5E7EB', marginBottom: '2.5rem',
          boxShadow: '0 4px 20px rgba(0,0,0,0.03)',
        }}>
          <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h3 className="syne" style={{ fontSize: '1.35rem', color: '#111827', marginBottom: '0.5rem' }}>
              Clinical Comparison View
            </h3>
            <p style={{ color: '#9CA3AF', fontSize: '0.85rem' }}>
              Analyse subtle morphological differences side-by-side.
            </p>
          </div>

          <div className="comparison-grid">
            {/* Nevus (Benign) */}
            <div className="comparison-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#111827' }}>Nevus (Benign)</span>
                <CheckCircle2 size={18} color="#10B981" />
              </div>
              <div style={{
                borderRadius: '12px', height: 120, overflow: 'hidden',
                position: 'relative',
              }}>
                <img
                  src={DISEASE_IMAGES['Melanocytic Nevus']}
                  alt="Nevus clinical reference"
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                />
              </div>
            </div>

            {/* Center swap icon */}
            <div style={{
              width: 44, height: 44, borderRadius: '50%',
              background: '#F3F4F6', border: '1px solid #E5E7EB',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              margin: '0 auto',
            }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6B7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M8 3L4 7l4 4" /><path d="M4 7h16" />
                <path d="M16 21l4-4-4-4" /><path d="M20 17H4" />
              </svg>
            </div>

            {/* Melanoma (Malignant) */}
            <div className="comparison-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#111827' }}>Melanoma (Malignant)</span>
                <AlertTriangle size={18} color="#EF4444" />
              </div>
              <div style={{
                borderRadius: '12px', height: 120, overflow: 'hidden',
                position: 'relative',
              }}>
                <img
                  src={DISEASE_IMAGES['Melanoma']}
                  alt="Melanoma clinical reference"
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
                />
              </div>
            </div>
          </div>

          <div style={{ textAlign: 'center', marginTop: '2rem' }}>
            <button
              onClick={() => navigate('/')}
              style={{
                background: '#111827', color: '#fff',
                border: 'none', borderRadius: '12px',
                padding: '14px 36px', fontSize: '0.82rem',
                fontWeight: 700, letterSpacing: '1.5px',
                textTransform: 'uppercase', cursor: 'pointer',
                transition: 'all 0.25s ease',
                boxShadow: '0 4px 14px rgba(0,0,0,0.15)',
              }}
              onMouseEnter={e => { e.currentTarget.style.background = '#EF4444'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(239,68,68,0.3)'; }}
              onMouseLeave={e => { e.currentTarget.style.background = '#111827'; e.currentTarget.style.boxShadow = '0 4px 14px rgba(0,0,0,0.15)'; }}
            >
              INITIATE ANALYSIS
            </button>
          </div>
        </div>

          </> /* end viewMode === 'case' */
        )}
        {/* ══════════════════════════════════════════ */}
        {selectedDisease && (
          <div
            onClick={() => setSelectedDisease(null)}
            style={{
              position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
              zIndex: 1000, background: 'rgba(15, 23, 42, 0.65)',
              backdropFilter: 'blur(8px)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              padding: '1.5rem',
            }}
          >
            <div
              onClick={e => e.stopPropagation()}
              style={{
                background: '#fff', borderRadius: '24px',
                padding: '2.5rem', maxWidth: 550, width: '100%',
                boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
                border: '1px solid rgba(229, 231, 235, 0.8)',
                position: 'relative',
                animation: 'fadeUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
                maxHeight: '90vh', overflowY: 'auto',
              }}
            >
              <button
                onClick={() => setSelectedDisease(null)}
                style={{
                  position: 'absolute', top: '1.25rem', right: '1.25rem',
                  background: '#F3F4F6', border: 'none', borderRadius: '50%',
                  width: 32, height: 32,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  cursor: 'pointer', color: '#4B5563',
                }}
              >
                ✕
              </button>

              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1rem' }}>
                <span style={{
                  background: `${selectedDisease.color}15`,
                  color: selectedDisease.color,
                  fontSize: '0.75rem', fontWeight: 800,
                  padding: '4px 12px', borderRadius: '999px',
                  textTransform: 'uppercase',
                }}>
                  {selectedDisease.risk} Risk
                </span>
                <span style={{ fontSize: '0.8rem', color: '#9CA3AF', fontWeight: 600 }}>ISIC Category</span>
              </div>

              <h2 className="syne" style={{ fontSize: '1.8rem', color: '#111827', marginBottom: '0.5rem' }}>
                {selectedDisease.name}
              </h2>
              <p style={{ color: '#4B5563', fontWeight: 600, fontSize: '0.95rem', marginBottom: '1.25rem' }}>
                {selectedDisease.short}
              </p>

              {/* Specimen preview */}
              <div style={{
                background: selectedDisease.gradient,
                borderRadius: '14px', height: 120, marginBottom: '1.25rem',
                position: 'relative', overflow: 'hidden',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <div style={{
                  position: 'absolute', inset: 0,
                  background: `radial-gradient(circle at 40% 50%, ${selectedDisease.color}25, transparent 60%)`,
                }} />
                <span style={{
                  fontSize: '0.7rem', fontWeight: 600, color: 'rgba(255,255,255,0.7)',
                  background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(8px)',
                  padding: '5px 14px', borderRadius: '999px', zIndex: 1,
                  border: '1px solid rgba(255,255,255,0.1)',
                }}>
                  Dermoscopic Specimen
                </span>
              </div>

              <div style={{
                background: '#F8FAFC', borderRadius: '16px', padding: '1.25rem',
                marginBottom: '1.5rem', border: '1px solid #E2E8F0',
              }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1E293B', marginBottom: '0.5rem' }}>
                  Pathophysiology & Description
                </h4>
                <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.65' }}>
                  {selectedDisease.details}
                </p>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1E293B', marginBottom: '0.75rem' }}>
                  Key Clinical Symptoms
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {selectedDisease.symptoms.map((s, idx) => (
                    <div key={idx} style={{
                      background: '#F9FAFB', padding: '8px 12px', borderRadius: '10px',
                      fontSize: '0.8rem', color: '#374151',
                      display: 'flex', alignItems: 'center', gap: '8px',
                    }}>
                      <CheckCircle2 size={14} color={selectedDisease.color} />
                      <span>{s}</span>
                    </div>
                  ))}
                </div>
              </div>

              {selectedDisease.clinicalObs && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1E293B', marginBottom: '0.75rem' }}>
                    Clinical Observations
                  </h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {selectedDisease.clinicalObs.map((obs, idx) => (
                      <li key={idx} style={{ fontSize: '0.82rem', color: '#374151', display: 'flex', alignItems: 'flex-start', gap: '8px' }}>
                        <span style={{ color: selectedDisease.color, fontWeight: 800, fontSize: '0.65rem', marginTop: '4px' }}>●</span>
                        {obs}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div style={{
                background: `${selectedDisease.color}10`,
                border: `1px solid ${selectedDisease.color}30`,
                borderRadius: '16px', padding: '1rem 1.25rem',
                display: 'flex', alignItems: 'center', gap: '12px',
              }}>
                <AlertCircle size={20} color={selectedDisease.color} />
                <div style={{ fontSize: '0.8rem', color: '#1F2937' }}>
                  <strong>Recommendation:</strong>{' '}
                  {selectedDisease.risk === 'Critical' || selectedDisease.risk === 'High'
                    ? 'Schedule a dermatoscopy consultation promptly.'
                    : 'Monitor regularly and follow standard annual skin screening routines.'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════ */}
        {/* ── DISCLAIMER ─────────────────────────── */}
        {/* ══════════════════════════════════════════ */}
        <div style={{
          background: '#fff', borderRadius: '24px', padding: '3rem',
          textAlign: 'center', border: '1px solid #E5E7EB',
        }}>
          <div style={{
            display: 'inline-flex', background: '#EFF6FF', padding: '1rem',
            borderRadius: '50%', color: '#2563EB', marginBottom: '1.5rem',
          }}>
            <Info size={32} />
          </div>
          <h2 className="syne" style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
            Always Consult a Professional
          </h2>
          <p style={{
            color: '#6B7280', maxWidth: '600px', margin: '0 auto',
            fontSize: '0.95rem', lineHeight: '1.7',
          }}>
            This guide is for informational purposes only. Skin disease identification requires
            professional clinical diagnosis. If you notice any suspicious changes in your skin,
            please see a dermatologist immediately.
          </p>
        </div>

      </div>
    </div>
  );
}
