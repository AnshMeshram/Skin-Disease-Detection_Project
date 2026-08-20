import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useReveal } from '../hooks/useReveal';
import FloatingParticlesBackground from './FloatingParticlesBackground';

const CLASSES = [
  { num: '01', name: 'Melanoma',                abbr: 'MEL'  },
  { num: '02', name: 'Melanocytic Nevus',        abbr: 'NV'   },
  { num: '03', name: 'Basal Cell Carcinoma',     abbr: 'BCC'  },
  { num: '04', name: 'Actinic Keratosis',        abbr: 'AK'   },
  { num: '05', name: 'Benign Keratosis',         abbr: 'BKL'  },
  { num: '06', name: 'Dermatofibroma',           abbr: 'DF'   },
  { num: '07', name: 'Vascular Lesion',          abbr: 'VASC' },
  { num: '08', name: 'Squamous Cell Carcinoma',  abbr: 'SCC'  },
  { num: '09', name: 'Healthy Skin',             abbr: 'OK'   },
];

export default function DiseaseGrid() {
  const ref = useReveal();
  const navigate = useNavigate();

  return (
    <section id="classes" className="reveal mesh-bg" ref={ref} style={{ padding: '6rem 2rem', position: 'relative', overflow: 'hidden' }}>
      <FloatingParticlesBackground variant="bubbles" count={28} colors={['#2563EB', '#38BDF8', '#10B981', '#6366F1', '#EC4899']} opacity={0.35} speed={0.4} />
      <div style={{ maxWidth: 1100, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <span style={{ background: '#fff', border: '1px solid #E5E7EB', borderRadius: 999, padding: '4px 14px', fontSize: '0.75rem', fontWeight: 500, color: '#4B5563', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>Classification</span>
          <h2 className="syne" style={{ fontSize: '1.75rem', fontWeight: 800, color: '#111827', marginTop: '1rem', marginBottom: '0.5rem', letterSpacing: '-0.01em' }}>9 Target Classes</h2>
          <p style={{ fontSize: '0.9375rem', color: '#6B7280', maxWidth: 600, margin: '0 auto' }}>8 ISIC 2019 disease categories plus a healthy skin class.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
          {CLASSES.map((c, i) => (
            <div key={c.abbr} 
            onClick={() => {
              const slug = c.name.toLowerCase().replace(/\s+/g, '-');
              navigate(`/guide#${slug}`);
            }}
            style={{
              background: '#fff', border: '1px solid #E5E7EB',
              borderRadius: 20, padding: '2rem 1.5rem', position: 'relative', overflow: 'hidden',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', cursor: 'pointer',
              boxShadow: '0 4px 20px rgba(0,0,0,0.04)',
              zIndex: 1,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              minHeight: 140
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#2563EB'; e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 10px 40px rgba(37,99,235,0.1)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#E5E7EB'; e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 20px rgba(0,0,0,0.04)'; }}
            >
              {/* Faded number */}
              <div className="syne" style={{ fontSize: '4rem', fontWeight: 900, color: 'rgba(37,99,235,0.06)', position: 'absolute', top: 10, right: 15, lineHeight: 1, userSelect: 'none', transition: 'all 0.3s' }}>{c.num}</div>
              <div className="syne" style={{ fontSize: '1.0625rem', fontWeight: 700, color: '#111827', marginBottom: 12, position: 'relative', zIndex: 2 }}>{c.name}</div>
              <div style={{ position: 'relative', zIndex: 2 }}>
                <span style={{ background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE', borderRadius: 999, padding: '4px 14px', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.05em' }}>{c.abbr}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
