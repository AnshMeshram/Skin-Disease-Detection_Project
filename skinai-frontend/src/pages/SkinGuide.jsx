import React from 'react';
import { useReveal } from '../hooks/useReveal';
import { Info, AlertCircle, CheckCircle2 } from 'lucide-react';

const DISEASES = [
  {
    name: 'Melanoma',
    short: 'The most serious type of skin cancer.',
    details: 'Develops in the pigment-producing melanocytes. It can spread to other organs if not caught early.',
    risk: 'Critical',
    symptoms: ['Asymmetrical shape', 'Irregular borders', 'Color variation', 'Diameter > 6mm'],
    color: '#EF4444'
  },
  {
    name: 'Melanocytic Nevus',
    short: 'Commonly known as a mole.',
    details: 'A benign (noncancerous) growth on the skin formed by a cluster of melanocytes.',
    risk: 'Benign',
    symptoms: ['Uniform tan or brown color', 'Round or oval shape', 'Distinct borders'],
    color: '#10B981'
  },
  {
    name: 'Basal Cell Carcinoma',
    short: 'Common, slow-growing skin cancer.',
    details: 'Occurs in basal cells which produce new skin cells. Rarely spreads but can be locally invasive.',
    risk: 'High',
    symptoms: ['Pearly or waxy bump', 'Flesh-colored lesion', 'Bleeding/scabby sore'],
    color: '#F59E0B'
  },
  {
    name: 'Actinic Keratosis',
    short: 'Precancerous scaly skin patch.',
    details: 'Caused by years of sun exposure. Can progress to Squamous Cell Carcinoma if untreated.',
    risk: 'Moderate',
    symptoms: ['Rough, dry patch', 'Itching or burning', 'Hard, wart-like surface'],
    color: '#3B82F6'
  },
  {
    name: 'Benign Keratosis',
    short: 'Non-cancerous skin growth.',
    details: 'Waxy, scaly, slightly elevated growths. Often appearing in older adults (Seborrheic Keratosis).',
    risk: 'Benign',
    symptoms: ['Waxy appearance', 'Brown, black or tan color', 'Stuck-on appearance'],
    color: '#6366F1'
  },
  {
    name: 'Dermatofibroma',
    short: 'Common benign fibrous nodule.',
    details: 'Often found on the legs. Usually develops after a minor injury like an insect bite.',
    risk: 'Benign',
    symptoms: ['Firm, small bump', 'Dimples when pinched', 'Variable color (pink to brown)'],
    color: '#8B5CF6'
  },
  {
    name: 'Vascular Lesion',
    short: 'Abnormal growth of blood vessels.',
    details: 'Includes cherry angiomas, hemangiomas, and port-wine stains. Usually harmless.',
    risk: 'Low',
    symptoms: ['Red, purple or blue color', 'Blanches on pressure', 'Soft or raised texture'],
    color: '#EC4899'
  },
  {
    name: 'Squamous Cell Carcinoma',
    short: 'Second most common skin cancer.',
    details: 'Develops in the squamous cells that make up the middle and outer layers of the skin.',
    risk: 'High',
    symptoms: ['Firm, red nodule', 'Flat sore with scaly crust', 'Non-healing ulcer'],
    color: '#B91C1C'
  },
  {
    name: 'Healthy Skin',
    short: 'Normal uninfected skin surface.',
    details: 'Skin with normal texture, pigmentation, and no signs of lesions or inflammation.',
    risk: 'None',
    symptoms: ['Smooth texture', 'Uniform pigmentation', 'No inflammation'],
    color: '#059669'
  }
];

export default function SkinGuide() {
  const ref = useReveal();

  React.useEffect(() => {
    const hash = window.location.hash;
    if (hash) {
      const id = hash.replace('#', '');
      const element = document.getElementById(id);
      if (element) {
        setTimeout(() => {
          element.scrollIntoView({ behavior: 'smooth' });
        }, 100);
      }
    }
  }, []);
  return (
    <div className="mesh-bg" style={{ minHeight: '100vh', padding: '100px 2rem 4rem' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto' }}>
        
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h1 className="syne" style={{ fontSize: '2.5rem', color: '#111827', marginBottom: '1rem' }}>Skin Disease Guide</h1>
          <p style={{ color: '#4B5563', maxWidth: '600px', margin: '0 auto' }}>
            Learn about common skin conditions, their symptoms, and risk levels to better understand your skin health.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
          {DISEASES.map((d, i) => (
            <div key={i} id={d.name.toLowerCase().replace(/\s+/g, '-')} className="reveal visible" style={{ 
              background: '#fff', 
              borderRadius: '20px', 
              padding: '2rem', 
              border: '1px solid #E5E7EB',
              boxShadow: '0 10px 15px -3px rgba(0,0,0,0.05)',
              display: 'flex',
              flexDirection: 'column',
              gap: '1.5rem',
              scrollMarginTop: '120px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2 className="syne" style={{ fontSize: '1.25rem', color: '#111827' }}>{d.name}</h2>
                <span style={{ 
                  background: `${d.color}15`, 
                  color: d.color, 
                  fontSize: '0.7rem', 
                  fontWeight: 700, 
                  padding: '4px 12px', 
                  borderRadius: '999px'
                }}>
                  Risk: {d.risk}
                </span>
              </div>

              <div>
                <p style={{ fontWeight: 600, fontSize: '0.9rem', color: '#374151', marginBottom: '0.5rem' }}>{d.short}</p>
                <p style={{ fontSize: '0.85rem', color: '#6B7280', lineHeight: '1.6' }}>{d.details}</p>
              </div>

              <div style={{ background: '#F9FAFB', borderRadius: '12px', padding: '1rem' }}>
                <p style={{ fontSize: '0.75rem', fontWeight: 700, color: '#9CA3AF', textTransform: 'uppercase', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <AlertCircle size={14} /> Key Symptoms
                </p>
                <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {d.symptoms.map((s, idx) => (
                    <li key={idx} style={{ fontSize: '0.8rem', color: '#4B5563', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <CheckCircle2 size={14} color={d.color} />
                      {s}
                    </li>
                  ))}
                </ul>
              </div>

              <button style={{ 
                marginTop: 'auto',
                background: 'transparent',
                border: `1px solid ${d.color}`,
                color: d.color,
                borderRadius: '999px',
                padding: '0.6rem',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
              onMouseEnter={e => { e.currentTarget.style.background = d.color; e.currentTarget.style.color = '#fff'; }}
              onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = d.color; }}
              >
                Learn More
              </button>
            </div>
          ))}
        </div>

        <div style={{ marginTop: '4rem', background: '#fff', borderRadius: '24px', padding: '3rem', textAlign: 'center', border: '1px solid #E5E7EB' }}>
           <div style={{ display: 'inline-flex', background: '#EFF6FF', padding: '1rem', borderRadius: '50%', color: '#2563EB', marginBottom: '1.5rem' }}>
              <Info size={32} />
           </div>
           <h2 className="syne" style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>Always Consult a Professional</h2>
           <p style={{ color: '#6B7280', maxWidth: '600px', margin: '0 auto', fontSize: '0.95rem', lineHeight: '1.7' }}>
             This guide is for informational purposes only. Skin disease identification requires professional clinical diagnosis. If you notice any suspicious changes in your skin, please see a dermatologist immediately.
           </p>
        </div>

      </div>
    </div>
  );
}
