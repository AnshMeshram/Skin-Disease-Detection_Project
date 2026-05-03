import { useReveal } from '../hooks/useReveal';

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
  return (
    <section id="classes" className="reveal" ref={ref} style={{ background: '#ECECEC', padding: '4rem 2rem' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <span style={{ background: '#fff', border: '1px solid #d1d5db', borderRadius: 999, padding: '4px 14px', fontSize: '0.72rem', fontWeight: 500, color: '#374151', boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>Classification</span>
          <h2 className="syne" style={{ fontSize: '1.75rem', fontWeight: 700, color: '#111827', marginTop: '0.6rem', marginBottom: '0.4rem' }}>9 Target Classes</h2>
          <p style={{ fontSize: '0.88rem', color: '#6b7280' }}>8 ISIC 2019 disease categories plus a healthy skin class.</p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: '0.85rem' }}>
          {CLASSES.map((c, i) => (
            <div key={c.abbr} style={{
              background: '#fff', border: '1px solid #e5e7eb',
              borderRadius: 12, padding: '1rem', position: 'relative', overflow: 'hidden',
              transition: 'all 0.2s ease', cursor: 'default',
              boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#93C5FD'; e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 6px 16px rgba(37,99,235,0.12)'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#e5e7eb'; e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)'; }}
            >
              {/* Faded number */}
              <div className="syne" style={{ fontSize: '2.8rem', fontWeight: 800, color: 'rgba(37,99,235,0.08)', position: 'absolute', top: 2, right: 8, lineHeight: 1, userSelect: 'none' }}>{c.num}</div>
              <div className="syne" style={{ fontSize: '0.88rem', fontWeight: 600, color: '#111827', marginBottom: 6 }}>{c.name}</div>
              <span style={{ background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE', borderRadius: 999, padding: '2px 9px', fontSize: '0.68rem', fontWeight: 600 }}>{c.abbr}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
