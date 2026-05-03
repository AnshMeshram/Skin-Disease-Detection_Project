import React from 'react';
import { CheckCircle2, Zap, Shield, BarChart3, Database, Maximize2, Cpu } from 'lucide-react';

const MODELS = [
  {
    name: 'EfficientNetB3',
    tag: 'BALANCED PICK',
    tagColor: '#10B981',
    description: 'Smart scaling - Compound efficiency',
    params: '~12M',
    accuracy: '81.6%',
    accWidth: '81.6%',
    resolution: '300 × 300',
    size: '~48 MB',
    speed: 'Fast',
    speedIcons: 3,
    memory: 'Low',
    memIcons: 1,
    specialized: 'General skin disease detection (Edge & resource-constrained)',
    advantages: [
      'Efficient compound scaling',
      'Excellent accuracy-to-compute ratio',
      'Strong transfer learning capability',
      'Ideal for mobile & edge deployment'
    ],
    fit: 'Edge Devices (Mobile / IoT)',
    fitWidth: '70%',
    score: '8.6',
    scoreText: 'Great Balance',
    scoreSub: 'Accuracy, Speed & Efficiency',
    borderColor: 'rgba(16,185,129,0.3)'
  },
  {
    name: 'InceptionV3',
    tag: 'MOST ESTABLISHED',
    tagColor: '#3B82F6',
    description: 'Multi-scale features - Proven medical AI workhorse',
    params: '~23M',
    accuracy: '78.8%',
    accWidth: '78.8%',
    resolution: '299 × 299',
    size: '~92 MB',
    speed: 'Moderate',
    speedIcons: 2,
    memory: 'Medium',
    memIcons: 2,
    specialized: 'Clinical-grade skin analysis (Research & Benchmarking)',
    advantages: [
      'Captures multi-scale textures (critical for skin)',
      'Factorized convolutions reduce computation',
      'Auxiliary classifiers reduce vanishing gradients',
      'Widely cited in dermatology AI research',
      'Robust fine-tuning on small datasets'
    ],
    fit: 'Research & Cloud (Higher Compute)',
    fitWidth: '50%',
    score: '8.9',
    scoreText: 'Most Reliable',
    scoreSub: 'Proven & Widely Trusted',
    borderColor: 'rgba(59,130,246,0.5)',
    highlight: true
  },
  {
    name: 'ConvNeXt Tiny',
    tag: 'MODERN ARCHITECTURE',
    tagColor: '#A855F7',
    description: 'Transformer-inspired - State-of-the-art CNN',
    params: '~28M',
    accuracy: '82.1%',
    accWidth: '82.1%',
    resolution: '224 × 224',
    size: '~109 MB',
    speed: 'Moderate',
    speedIcons: 2,
    memory: 'Medium-High',
    memIcons: 3,
    specialized: 'High-accuracy skin diagnosis (Advanced Clinical Use)',
    advantages: [
      'Depthwise 7x7 convolutions mimic ViT',
      'LayerNorm + GELU for stable training',
      'Highest raw accuracy among the three',
      'Excels with diverse skin datasets',
      'Benefits greatly from longer training'
    ],
    fit: 'Cloud / GPU Server (Production Grade)',
    fitWidth: '90%',
    score: '9.2',
    scoreText: 'Best Accuracy',
    scoreSub: 'Highest Performance Ceiling',
    borderColor: 'rgba(168,85,247,0.3)'
  }
];

const FEATURES = [
  { name: 'Model Highlights', icon: <Shield size={16} /> },
  { name: 'Parameters', icon: <Database size={16} /> },
  { name: 'Top-1 Accuracy', icon: <BarChart3 size={16} />, sub: '(ImageNet Pretrained)' },
  { name: 'Input Resolution', icon: <Maximize2 size={16} />, sub: '(Recommended)' },
  { name: 'Model Size', icon: <Database size={16} />, sub: '(On Disk)' },
  { name: 'Inference Speed', icon: <Zap size={16} />, sub: '(Relative)' },
  { name: 'Memory Footprint', icon: <Cpu size={16} />, sub: '(GPU Inference)' },
  { name: 'Specialized For', icon: <Shield size={16} />, sub: '(Skin Disease Detection)' },
  { name: 'Key Advantages', icon: <Zap size={16} /> },
  { name: 'Deployment Fit', icon: <Database size={16} /> },
  { name: 'Overall Score', icon: <Shield size={16} /> }
];

export default function ModelComparisonTable() {
  return (
    <div style={{ background: '#04101E', color: '#fff', padding: '2rem', borderRadius: '12px', border: '1px solid rgba(14,165,233,0.2)' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr 1fr 1fr', gap: '1rem' }}>
        
        {/* Sidebar labels */}
        <div style={{ paddingTop: '100px' }}>
          <h4 className="syne" style={{ fontSize: '0.75rem', color: '#94A3B8', textTransform: 'uppercase', marginBottom: '2rem' }}>Model Overview</h4>
          {FEATURES.map((f, i) => (
            <div key={i} style={{ marginBottom: f.name === 'Key Advantages' ? '180px' : f.name === 'Specialized For' ? '40px' : '32px', height: '40px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#EFF6FF' }}>
                <span style={{ color: '#0EA5E9' }}>{f.icon}</span>
                <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{f.name}</span>
              </div>
              {f.sub && <div style={{ fontSize: '0.65rem', color: '#94A3B8', marginLeft: '24px' }}>{f.sub}</div>}
            </div>
          ))}
        </div>

        {/* Model Columns */}
        {MODELS.map((m, i) => (
          <div key={i} style={{ 
            background: 'rgba(8,24,40,0.5)', 
            border: `1px solid ${m.borderColor}`, 
            borderRadius: '16px', 
            padding: '1.5rem',
            position: 'relative',
            boxShadow: m.highlight ? '0 0 30px rgba(59,130,246,0.15)' : 'none'
          }}>
            {/* Header */}
            <div style={{ marginBottom: '1.5rem' }}>
              <div style={{ 
                background: `${m.tagColor}20`, 
                color: m.tagColor, 
                fontSize: '0.65rem', 
                fontWeight: 700, 
                padding: '4px 8px', 
                borderRadius: '4px', 
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                marginBottom: '0.75rem'
              }}>
                <CheckCircle2 size={10} /> {m.tag}
              </div>
              <h3 className="syne" style={{ fontSize: '1.25rem', marginBottom: '0.25rem' }}>{m.name}</h3>
              <p style={{ fontSize: '0.7rem', color: '#94A3B8' }}>{m.description}</p>
            </div>

            {/* Stats */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
              <div style={{ height: '40px', fontSize: '0.9rem', fontWeight: 600 }}>{m.params}</div>
              
              <div style={{ height: '40px' }}>
                <div style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '4px' }}>{m.accuracy}</div>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                  <div style={{ height: '100%', width: m.accWidth, background: m.tagColor, borderRadius: '2px' }} />
                </div>
              </div>

              <div style={{ height: '40px', fontSize: '0.9rem', fontWeight: 600 }}>{m.resolution}</div>
              <div style={{ height: '40px', fontSize: '0.9rem', fontWeight: 600 }}>{m.size}</div>

              <div style={{ height: '40px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.8rem' }}>{m.speed}</span>
                <div style={{ display: 'flex', gap: '2px' }}>
                  {[1,2,3,4].map(idx => (
                    <Zap key={idx} size={12} fill={idx <= m.speedIcons ? m.tagColor : 'none'} color={idx <= m.speedIcons ? m.tagColor : 'rgba(255,255,255,0.1)'} />
                  ))}
                </div>
              </div>

              <div style={{ height: '40px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '0.8rem' }}>{m.memory}</span>
                <div style={{ display: 'flex', gap: '2px' }}>
                  {[1,2,3,4].map(idx => (
                    <Cpu key={idx} size={12} color={idx <= m.memIcons ? m.tagColor : 'rgba(255,255,255,0.1)'} />
                  ))}
                </div>
              </div>

              <div style={{ height: '40px', fontSize: '0.75rem', lineHeight: '1.4', color: '#CBD5E1' }}>
                {m.specialized}
              </div>

              <div style={{ minHeight: '180px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {m.advantages.map((adv, idx) => (
                  <div key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '0.7rem', color: '#94A3B8' }}>
                    <CheckCircle2 size={12} color={m.tagColor} style={{ marginTop: '2px' }} />
                    <span>{adv}</span>
                  </div>
                ))}
              </div>

              <div style={{ height: '40px' }}>
                <div style={{ fontSize: '0.75rem', color: '#EFF6FF', marginBottom: '4px' }}>{m.fit}</div>
                <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', borderRadius: '2px' }}>
                  <div style={{ height: '100%', width: m.fitWidth, background: m.tagColor, borderRadius: '2px' }} />
                </div>
              </div>

              <div style={{ height: '40px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ 
                  width: '40px', height: '40px', borderRadius: '50%', border: `2px solid ${m.tagColor}`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '0.9rem'
                }}>
                  {m.score}
                </div>
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: m.tagColor }}>{m.scoreText}</div>
                  <div style={{ fontSize: '0.6rem', color: '#94A3B8' }}>{m.scoreSub}</div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
