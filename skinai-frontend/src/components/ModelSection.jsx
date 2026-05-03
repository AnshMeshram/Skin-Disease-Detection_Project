import { useState } from 'react';
import { useReveal } from '../hooks/useReveal';

const MODELS = [
  {
    id: 'efficientnet', label: 'EfficientNet-B3',
    stats: [{ val: '~12M', label: 'Parameters' }, { val: '300×300', label: 'Input Size' }, { val: '3D-CA', label: 'Attention' }, { val: '9', label: 'Classes' }],
    desc: 'EfficientNet-B3 applies compound scaling to jointly scale depth, width, and resolution. Enhanced with a 3D convolutional soft-attention module inserted before the classifier head. Trained with Focal Loss + label smoothing and AdamW with OneCycleLR scheduler.',
    tags: ['AdamW', 'OneCycleLR', 'Focal Loss', 'AMP', 'ImageNet Pretrained'],
  },
  {
    id: 'inception', label: 'Inception V3',
    stats: [{ val: '~24M', label: 'Parameters' }, { val: '299×299', label: 'Input Size' }, { val: '3D-CA', label: 'Attention' }, { val: '5-Fold', label: 'Cross-Val' }],
    desc: 'Inception V3 uses factorized convolutions and auxiliary classifiers to improve gradient flow through deep networks. Multi-scale inception modules capture dermoscopic patterns at different spatial frequencies. Pretrained on ImageNet.',
    tags: ['Factorized Conv', 'Auxiliary Loss', 'Multi-Scale', 'Label Smoothing'],
  },
  {
    id: 'convnext', label: 'ConvNeXt Tiny',
    stats: [{ val: '~28M', label: 'Parameters' }, { val: '224×224', label: 'Input Size' }, { val: '3D-CA', label: 'Attention' }, { val: 'AdamW', label: 'Optimizer' }],
    desc: 'ConvNeXt Tiny modernizes classical ConvNets with transformer-inspired design: depthwise convolutions, inverted bottlenecks, and LayerNorm. Strong inductive biases and competitive accuracy make it an excellent ensemble member.',
    tags: ['Depthwise Conv', 'LayerNorm', 'Class Weights', 'Grad-CAM'],
  },
];

export default function ModelSection() {
  const [active, setActive] = useState(0);
  const ref = useReveal();
  const m = MODELS[active];

  return (
    <section id="model" className="reveal" ref={ref} style={{ background: '#fff', padding: '4rem 2rem' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <span style={{
            background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE',
            borderRadius: 999, padding: '4px 14px', fontSize: '0.72rem', fontWeight: 600,
            textTransform: 'uppercase', letterSpacing: '0.06em',
          }}>Architecture</span>
          <h2 className="syne" style={{ fontSize: '1.75rem', fontWeight: 700, color: '#111827', marginTop: '0.6rem', marginBottom: '0.4rem' }}>Model Details</h2>
          <p style={{ fontSize: '0.9rem', color: '#6b7280' }}>Three architectures trained on ISIC 2019 with 5-fold cross-validation.</p>
        </div>

        {/* Tab switcher */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', background: '#f3f4f6', borderRadius: 10, padding: 4, gap: 2 }}>
            {MODELS.map((mod, i) => (
              <button key={mod.id} onClick={() => setActive(i)} style={{
                background: active === i ? '#fff' : 'transparent',
                color: active === i ? '#111827' : '#6b7280',
                border: 'none', borderRadius: 8,
                padding: '7px 18px', fontSize: '0.82rem', fontWeight: 500,
                cursor: 'pointer', transition: 'all 0.2s ease',
                boxShadow: active === i ? '0 1px 3px rgba(0,0,0,0.1)' : 'none',
              }}>{mod.label}</button>
            ))}
          </div>
        </div>

        {/* Stat grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: '0.75rem', marginBottom: '1rem' }}>
          {m.stats.map(s => (
            <div key={s.label} style={{ background: '#F9FAFB', border: '1px solid #e5e7eb', borderRadius: 10, padding: '1rem', textAlign: 'center' }}>
              <div className="syne" style={{ fontSize: '1.3rem', fontWeight: 700, color: '#2563EB' }}>{s.val}</div>
              <div style={{ fontSize: '0.72rem', color: '#6b7280', marginTop: 4 }}>{s.label}</div>
            </div>
          ))}
        </div>

        {/* Detail */}
        <div style={{ background: '#F9FAFB', border: '1px solid #e5e7eb', borderRadius: 12, padding: '1.25rem 1.5rem' }}>
          <h3 className="syne" style={{ fontWeight: 600, marginBottom: '0.6rem', color: '#111827' }}>Architecture Overview</h3>
          <p style={{ fontSize: '0.88rem', color: '#4b5563', lineHeight: 1.75, marginBottom: '0.9rem' }}>{m.desc}</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {m.tags.map(t => (
              <span key={t} style={{
                background: '#EFF6FF', color: '#2563EB', border: '1px solid #BFDBFE',
                borderRadius: 999, padding: '3px 10px', fontSize: '0.72rem', fontWeight: 500,
              }}>{t}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
