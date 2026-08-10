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
    <section id="classes" className="reveal mesh-bg" ref={ref} style={{ padding: '8rem 2rem', position: 'relative', overflow: 'hidden' }}>
      <FloatingParticlesBackground count={24} colors={['#3B82F6', '#60A5FA', '#2DD4BF', '#93C5FD']} opacity={0.15} speed={0.4} />
      <div style={{ maxWidth: 1000, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
          <span style={{ background: '#fff', border: '1px solid #d1d5db', borderRadius: 999, padding: '4px 14px', fontSize: '0.72rem', fontWeight: 500, color: '#374151', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>Classification</span>
          <h2 className="syne" style={{ fontSize: '2.5rem', fontWeight: 800, color: '#111827', marginTop: '1rem', marginBottom: '0.5rem', letterSpacing: '-0.02em' }}>9 Target Classes</h2>
          <p style={{ fontSize: '1rem', color: '#6b7280', maxWidth: 600, margin: '0 auto' }}>8 ISIC 2019 disease categories plus a healthy skin class.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
          {CLASSES.map((c, i) => (
            <div key={c.abbr} 
            onClick={() => {
              const slug = c.name.toLowerCase().replace(/\s+/g, '-');
              navigate(`/guide#${slug}`);
            }}
            style={{
              background: '#fff', border: '1px solid #e5e7eb',
              borderRadius: 20, padding: '2rem 1.5rem', position: 'relative', overflow: 'hidden',
              transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)', cursor: 'pointer',
              boxShadow: '0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01)',
              zIndex: 1,
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
              minHeight: 140
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#3B82F6'; e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.boxShadow = '0 20px 25px -5px rgba(59,130,246,0.1), 0 10px 10px -5px rgba(59,130,246,0.04)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.01)'; }}
            >
              {/* Faded number */}
              <div className="syne" style={{ fontSize: '4rem', fontWeight: 900, color: 'rgba(37,99,235,0.06)', position: 'absolute', top: 10, right: 15, lineHeight: 1, userSelect: 'none', transition: 'all 0.3s' }}>{c.num}</div>
              <div className="syne" style={{ fontSize: '1.1rem', fontWeight: 700, color: '#111827', marginBottom: 12, position: 'relative', zIndex: 2 }}>{c.name}</div>
              <div style={{ position: 'relative', zIndex: 2 }}>
                <span style={{ background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE', borderRadius: 999, padding: '4px 14px', fontSize: '0.72rem', fontWeight: 700, letterSpacing: '0.05em' }}>{c.abbr}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
