import React from 'react';
import { CheckCircle2, Zap, Shield, BarChart3, Database, Maximize2, Cpu, Star } from 'lucide-react';

const MODELS = [
  {
    name: 'EfficientNetB3',
    tag: 'BEST PERFORMER',
    tagColor: '#10B981',
    description: 'Highest local accuracy',
    params: '~12M',
    accuracy: '90.2%',
    accWidth: '90.2%',
    resolution: '300 × 300',
    size: '~48 MB',
    speed: 'Fast',
    speedIcons: 3,
    memory: 'Low',
    memIcons: 1,
    specialized: 'General skin disease detection',
    advantages: [
      'Efficient scaling',
      'High compute ratio',
      'Strong transfer learning',
      'Ideal for edge devices'
    ],
    fit: 'Edge Devices',
    fitWidth: '70%',
    score: '9.5',
    scoreText: 'Best Accuracy',
    scoreSub: 'Highest Performance',
    borderColor: 'rgba(16,185,129,0.5)',
    highlight: true
  },
  {
    name: 'InceptionV3',
    tag: 'MOST ESTABLISHED',
    tagColor: '#3B82F6',
    description: 'Proven medical workhorse',
    params: '~23M',
    accuracy: '85.0%',
    accWidth: '85.0%',
    resolution: '299 × 299',
    size: '~92 MB',
    speed: 'Moderate',
    speedIcons: 2,
    memory: 'Medium',
    memIcons: 2,
    specialized: 'Clinical-grade analysis',
    advantages: [
      'Multi-scale textures',
      'Factorized convolutions',
      'Auxiliary classifiers',
      'Widely cited research'
    ],
    fit: 'Cloud / Research',
    fitWidth: '50%',
    score: '8.8',
    scoreText: 'Highly Reliable',
    scoreSub: 'Proven & Trusted',
    borderColor: 'rgba(59,130,246,0.3)',
    highlight: false
  },
  {
    name: 'ConvNeXt Tiny',
    tag: 'MODERN ARCHITECTURE',
    tagColor: '#A855F7',
    description: 'Transformer-inspired CNN',
    params: '~28M',
    accuracy: '88.0%',
    accWidth: '88.0%',
    resolution: '224 × 224',
    size: '~109 MB',
    speed: 'Moderate',
    speedIcons: 2,
    memory: 'Medium-High',
    memIcons: 3,
    specialized: 'Advanced clinical diagnosis',
    advantages: [
      'Depthwise 7x7 convs',
      'LayerNorm + GELU',
      'ViT-like performance',
      'Modern training loop'
    ],
    fit: 'Production GPU',
    fitWidth: '90%',
    score: '9.2',
    scoreText: 'Excellent Accuracy',
    scoreSub: 'High Ceiling',
    borderColor: 'rgba(168,85,247,0.3)',
    highlight: false
  }
];

const Row = ({ label, icon, sub, children }) => (
  <>
    <div style={{ padding: '1.25rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#0EA5E9' }}>
        {icon}
        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#EFF6FF' }}>{label}</span>
      </div>
      {sub && <span style={{ fontSize: '0.65rem', color: '#94A3B8', marginLeft: '24px' }}>{sub}</span>}
    </div>
    {children}
  </>
);

export default function ModelComparisonTable() {
  return (
    <div style={{ 
      background: '#04101E', 
      color: '#fff', 
      padding: '2rem', 
      borderRadius: '24px', 
      border: '1px solid rgba(14,165,233,0.2)',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
    }}>
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: '220px repeat(3, 1fr)', 
        alignItems: 'stretch'
      }}>
        
        {/* Header Row */}
        <div style={{ paddingBottom: '2rem' }}>
           <h4 className="syne" style={{ fontSize: '0.75rem', color: '#0EA5E9', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Comparison Matrix</h4>
        </div>
        {MODELS.map((m, i) => (
          <div key={i} style={{ 
            padding: '0 1.5rem 2rem', 
            borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)',
            position: 'relative'
          }}>
            {m.highlight && (
              <div style={{ position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)', background: m.tagColor, color: '#fff', fontSize: '0.6rem', fontWeight: 900, padding: '2px 8px', borderRadius: '4px', textTransform: 'uppercase', zIndex: 10 }}>
                Recommendation
              </div>
            )}
            <div style={{ 
              background: `${m.tagColor}15`, 
              color: m.tagColor, 
              fontSize: '0.65rem', 
              fontWeight: 800, 
              padding: '4px 10px', 
              borderRadius: '6px', 
              display: 'inline-block',
              marginBottom: '1rem'
            }}>
              {m.tag}
            </div>
            <h3 className="syne" style={{ fontSize: '1.25rem', marginBottom: '0.25rem', color: '#fff' }}>{m.name}</h3>
            <p style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{m.description}</p>
          </div>
        ))}

        {/* Accuracy Row */}
        <Row label="Top-1 Accuracy" icon={<BarChart3 size={16} />} sub="Best Fold Performance">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ fontSize: '1.1rem', fontWeight: 800, color: m.tagColor, marginBottom: '6px' }}>{m.accuracy}</div>
              <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ height: '100%', width: m.accWidth, background: m.tagColor, borderRadius: '2px' }} />
              </div>
            </div>
          ))}
        </Row>

        {/* Parameters Row */}
        <Row label="Parameters" icon={<Database size={16} />} sub="Model Complexity">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{m.params}</span>
            </div>
          ))}
        </Row>

        {/* Resolution Row */}
        <Row label="Resolution" icon={<Maximize2 size={16} />} sub="Input Image Size">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{m.resolution}</span>
            </div>
          ))}
        </Row>

        {/* Speed Row */}
        <Row label="Inference Speed" icon={<Zap size={16} />} sub="Latency Profile">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{m.speed}</span>
              <div style={{ display: 'flex', gap: '2px' }}>
                {[1,2,3].map(idx => (
                  <Zap key={idx} size={10} fill={idx <= m.speedIcons ? m.tagColor : 'none'} color={idx <= m.speedIcons ? m.tagColor : 'rgba(255,255,255,0.1)'} />
                ))}
              </div>
            </div>
          ))}
        </Row>

        {/* Memory Row */}
        <Row label="Memory Usage" icon={<Cpu size={16} />} sub="GPU VRAM Usage">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{m.memory}</span>
              <div style={{ display: 'flex', gap: '2px' }}>
                {[1,2,3].map(idx => (
                  <Cpu key={idx} size={10} fill={idx <= m.memIcons ? m.tagColor : 'none'} color={idx <= m.memIcons ? m.tagColor : 'rgba(255,255,255,0.1)'} />
                ))}
              </div>
            </div>
          ))}
        </Row>

        {/* Advantages Row */}
        <Row label="Key Advantages" icon={<CheckCircle2 size={16} />} sub="Architectural Strengths">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid rgba(255,255,255,0.05)', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {m.advantages.map((adv, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: m.tagColor }} />
                    <span style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{adv}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </Row>

        {/* Overall Score Row */}
        <Row label="Clinical Score" icon={<Shield size={16} />} sub="Weighted Evaluation">
          {MODELS.map((m, i) => (
            <div key={i} style={{ padding: '1.25rem 1.5rem', borderLeft: i === 0 ? 'none' : '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <div style={{ 
                width: '38px', height: '38px', borderRadius: '10px', background: `${m.tagColor}20`, border: `1px solid ${m.tagColor}40`,
                display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 900, fontSize: '0.9rem', color: m.tagColor
              }}>
                {m.score}
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, color: '#fff' }}>{m.scoreText}</span>
                <span style={{ fontSize: '0.6rem', color: '#94A3B8' }}>{m.scoreSub}</span>
              </div>
            </div>
          ))}
        </Row>

      </div>
    </div>
  );
}
