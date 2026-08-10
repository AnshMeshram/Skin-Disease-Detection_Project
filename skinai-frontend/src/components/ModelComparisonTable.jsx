import React from 'react';
import { CheckCircle2, Zap, Shield, BarChart3, Database, Maximize2, Cpu, Award } from 'lucide-react';

const MODELS = [
  {
    name: 'EfficientNet-B3',
    tag: 'RECOMMENDED',
    tagBg: 'rgba(16, 185, 129, 0.15)',
    tagColor: '#34D399',
    description: 'Optimal balance of computational efficiency and accuracy',
    params: '~12.2 Million',
    accuracy: '90.2%',
    accWidth: '90.2%',
    resolution: '300 × 300 px',
    size: '~48 MB',
    speed: 'High (Fast)',
    speedLevel: 3,
    memory: 'Low (~1.2 GB VRAM)',
    advantages: [
      'Compound feature scaling',
      'Superior accuracy-to-param ratio',
      'Fast edge & cloud inference',
      'Strong dermoscopic transfer'
    ],
    score: '9.5',
    scoreText: 'Top Performer',
    highlight: true
  },
  {
    name: 'Inception-V3',
    tag: 'ESTABLISHED WORKHORSE',
    tagBg: 'rgba(59, 130, 246, 0.15)',
    tagColor: '#60A5FA',
    description: 'Classic multi-scale factorized convolution baseline',
    params: '~23.8 Million',
    accuracy: '85.0%',
    accWidth: '85.0%',
    resolution: '299 × 299 px',
    size: '~92 MB',
    speed: 'Moderate',
    speedLevel: 2,
    memory: 'Medium (~2.1 GB VRAM)',
    advantages: [
      'Multi-scale texture filters',
      'Factorized 7×7 convolutions',
      'Extensive clinical literature',
      'Highly stable convergence'
    ],
    score: '8.8',
    scoreText: 'Clinical Baseline',
    highlight: false
  },
  {
    name: 'ConvNeXt-Tiny',
    tag: 'MODERN ARCHITECTURE',
    tagBg: 'rgba(168, 85, 247, 0.15)',
    tagColor: '#C084FC',
    description: 'Pure CNN redesigned with modern Vision Transformer principles',
    params: '~28.6 Million',
    accuracy: '88.0%',
    accWidth: '88.0%',
    resolution: '224 × 224 px',
    size: '~109 MB',
    speed: 'Moderate',
    speedLevel: 2,
    memory: 'Medium-High (~2.8 GB VRAM)',
    advantages: [
      'Depthwise 7×7 convolutions',
      'GELU activations & LayerNorm',
      'High structural capacity',
      'Robust feature representations'
    ],
    score: '9.2',
    scoreText: 'High Capacity',
    highlight: false
  }
];

const MatrixRow = ({ label, icon, sub, children }) => (
  <>
    <div style={{
      padding: '1.25rem 1rem',
      borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center',
      background: 'rgba(15, 23, 42, 0.6)'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#38BDF8' }}>
        {icon}
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F1F5F9', whiteSpace: 'normal', wordBreak: 'break-word' }}>
          {label}
        </span>
      </div>
      {sub && (
        <span style={{ fontSize: '0.7rem', color: '#94A3B8', marginLeft: '24px', marginTop: '2px', whiteSpace: 'normal' }}>
          {sub}
        </span>
      )}
    </div>
    {children}
  </>
);

export default function ModelComparisonTable() {
  return (
    <div style={{
      background: '#0B0F19',
      color: '#F8FAFC',
      borderRadius: '24px',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.6)',
      overflow: 'hidden',
    }}>
      {/* Scrollable Container */}
      <div className="table-responsive-wrapper" style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: '220px repeat(3, minmax(260px, 1fr))',
          alignItems: 'stretch',
          minWidth: '1000px',
        }}>

          {/* Header Row */}
          <div style={{ padding: '1.75rem 1rem', background: '#0F172A', borderBottom: '2px solid rgba(255, 255, 255, 0.1)' }}>
            <span style={{
              fontSize: '0.7rem',
              fontWeight: 800,
              color: '#38BDF8',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              display: 'inline-block',
              marginBottom: '4px'
            }}>
              BENCHMARK MATRIX
            </span>
            <h3 className="syne" style={{ fontSize: '1.1rem', color: '#F8FAFC', margin: 0 }}>
              Architecture Evaluation
            </h3>
          </div>

          {MODELS.map((m, i) => (
            <div key={i} style={{
              padding: '1.75rem 1.25rem',
              background: m.highlight ? 'rgba(16, 185, 129, 0.06)' : '#0F172A',
              borderBottom: '2px solid rgba(255, 255, 255, 0.1)',
              borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
              position: 'relative'
            }}>
              {m.highlight && (
                <div style={{
                  position: 'absolute', top: 10, right: 14,
                  background: '#059669', color: '#fff',
                  fontSize: '0.62rem', fontWeight: 800,
                  padding: '3px 8px', borderRadius: '999px',
                  letterSpacing: '0.05em', textTransform: 'uppercase',
                  display: 'flex', alignItems: 'center', gap: '4px'
                }}>
                  <Award size={12} /> Top Pick
                </div>
              )}
              <span style={{
                background: m.tagBg,
                color: m.tagColor,
                fontSize: '0.68rem',
                fontWeight: 800,
                padding: '4px 10px',
                borderRadius: '6px',
                display: 'inline-block',
                marginBottom: '0.6rem',
                letterSpacing: '0.03em'
              }}>
                {m.tag}
              </span>
              <h3 className="syne" style={{ fontSize: '1.25rem', color: '#F8FAFC', marginBottom: '0.35rem' }}>
                {m.name}
              </h3>
              <p style={{ fontSize: '0.78rem', color: '#94A3B8', lineHeight: 1.4, margin: 0 }}>
                {m.description}
              </p>
            </div>
          ))}

          {/* Accuracy Row */}
          <MatrixRow label="Validation Accuracy" icon={<BarChart3 size={16} />} sub="5-Fold Cross Validation">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19'
              }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 800, color: m.tagColor, marginBottom: '6px' }}>
                  {m.accuracy}
                </div>
                <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: m.accWidth, background: m.tagColor, borderRadius: '3px' }} />
                </div>
              </div>
            ))}
          </MatrixRow>

          {/* Parameters Row */}
          <MatrixRow label="Parameters" icon={<Database size={16} />} sub="Model Capacity & Size">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19',
                display: 'flex', flexDirection: 'column', justifyContent: 'center'
              }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#F1F5F9' }}>{m.params}</span>
                <span style={{ fontSize: '0.72rem', color: '#94A3B8' }}>Weight file {m.size}</span>
              </div>
            ))}
          </MatrixRow>

          {/* Input Resolution Row */}
          <MatrixRow label="Input Resolution" icon={<Maximize2 size={16} />} sub="Dermoscopic Spatial Dimensions">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19',
                display: 'flex', alignItems: 'center'
              }}>
                <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#F1F5F9' }}>{m.resolution}</span>
              </div>
            ))}
          </MatrixRow>

          {/* Inference Speed Row */}
          <MatrixRow label="Inference Speed" icon={<Zap size={16} />} sub="Latency & Throughput Profile">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19',
                display: 'flex', alignItems: 'center', gap: '8px'
              }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#F1F5F9' }}>{m.speed}</span>
                <div style={{ display: 'flex', gap: '3px' }}>
                  {[1, 2, 3].map(idx => (
                    <Zap key={idx} size={12} fill={idx <= m.speedLevel ? m.tagColor : 'none'} color={idx <= m.speedLevel ? m.tagColor : 'rgba(255, 255, 255, 0.2)'} />
                  ))}
                </div>
              </div>
            ))}
          </MatrixRow>

          {/* Memory VRAM Row */}
          <MatrixRow label="GPU VRAM Memory" icon={<Cpu size={16} />} sub="Inference Memory Footprint">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19',
                display: 'flex', alignItems: 'center'
              }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#CBD5E1' }}>{m.memory}</span>
              </div>
            ))}
          </MatrixRow>

          {/* Key Advantages Row */}
          <MatrixRow label="Architectural Advantages" icon={<CheckCircle2 size={16} />} sub="Key Structural Highlights">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.07)',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19'
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  {m.advantages.map((adv, idx) => (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <CheckCircle2 size={13} color={m.tagColor} style={{ flexShrink: 0 }} />
                      <span style={{ fontSize: '0.78rem', color: '#94A3B8', lineHeight: 1.3 }}>{adv}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </MatrixRow>

          {/* Overall Clinical Score Row */}
          <MatrixRow label="Overall Rating" icon={<Shield size={16} />} sub="Weighted Medical Benchmark">
            {MODELS.map((m, i) => (
              <div key={i} style={{
                padding: '1.25rem 1.25rem',
                borderLeft: '1px solid rgba(255, 255, 255, 0.08)',
                background: '#0B0F19',
                display: 'flex', alignItems: 'center', gap: '12px'
              }}>
                <div style={{
                  width: '42px', height: '42px', borderRadius: '12px',
                  background: m.tagBg, border: `1px solid ${m.tagColor}40`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontWeight: 900, fontSize: '1rem', color: m.tagColor, flexShrink: 0
                }}>
                  {m.score}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '0.82rem', fontWeight: 800, color: '#F8FAFC' }}>{m.scoreText}</span>
                  <span style={{ fontSize: '0.68rem', color: '#94A3B8' }}>Score out of 10</span>
                </div>
              </div>
            ))}
          </MatrixRow>

        </div>
      </div>
    </div>
  );
}
