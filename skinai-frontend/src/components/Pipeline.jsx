import React from 'react';
import { useReveal } from '../hooks/useReveal';

const STEPS = [
  { key: 'original',       label: 'White Balance', desc: 'Color correction'  },
  { key: 'step1_resized',  label: 'Original',      desc: 'Raw input'         },
  { key: 'step5_no_hair',  label: 'Hair removed',  desc: 'BlackHat + Telea'  },
  { key: 'step4_clahe',    label: 'ITA CLAHE',     desc: 'LAB contrast'      },
  { key: 'step2_denoised', label: 'Skin Mask',     desc: 'Segmentation'      },
  { key: 'step3_lab',      label: 'Lesion Region', desc: 'Final crop'        },
];

function StepCard({ step, index, imgs, activeStep }) {
  const imgData = imgs?.[step.key];
  const done    = imgs && activeStep > index;
  const active  = activeStep === index && !!imgs;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <div style={{
        width: 90, height: 90,
        borderRadius: 10,
        border: `2px solid ${active ? '#2563EB' : done ? '#10B981' : '#d1d5db'}`,
        background: '#fff',
        overflow: 'hidden',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        boxShadow: active ? '0 0 12px rgba(37,99,235,0.3)' : done ? '0 0 8px rgba(16,185,129,0.2)' : '0 1px 3px rgba(0,0,0,0.08)',
        transition: 'all 0.35s ease',
        flexShrink: 0,
      }}>
        {imgData ? (
          <img
            src={`data:image/png;base64,${imgData}`}
            alt={step.label}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
          />
        ) : (
          <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f9fafb' }}>
             <div className="spinner" style={{ borderTopColor: '#2563EB', opacity: active ? 1 : 0 }} />
          </div>
        )}
      </div>
      <span style={{ fontSize: '0.72rem', fontWeight: 600, color: active ? '#2563EB' : done ? '#10B981' : '#374151', textAlign: 'center', fontFamily: 'Syne, sans-serif' }}>
        {step.label}
      </span>
    </div>
  );
}

export default function Pipeline({ pipelineResult, activeStep, isVisible }) {
  const ref = useReveal();

  if (!isVisible) return null;

  return (
    <section id="preprocessing" className="reveal visible liquid-bg" ref={ref}
      style={{ padding: '6rem 2rem', animation: 'fadeUp 0.5s ease', position: 'relative', overflow: 'hidden' }}
    >
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <span style={{
            background: activeStep >= 6 ? '#ECFDF5' : '#fff', 
            border: `1px solid ${activeStep >= 6 ? '#10B981' : '#d1d5db'}`,
            borderRadius: 999, padding: '5px 18px',
            fontSize: '0.78rem', fontWeight: 600, 
            color: activeStep >= 6 ? '#10B981' : '#374151',
            boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            display: 'flex', alignItems: 'center', gap: '8px',
            transition: 'all 0.3s ease'
          }}>
            {activeStep >= 6 ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
            ) : (
              <span className="spinner" style={{ width: '10px', height: '10px', borderTopColor: '#2563EB' }} />
            )}
            {activeStep >= 6 ? 'Pipeline Complete' : 'Processing Pipeline...'}
          </span>
        </div>

        <div style={{
          background: '#fff', border: '1px solid #e5e7eb',
          borderRadius: 16, padding: '1.5rem 1.75rem',
          boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
        }}>
          <div style={{
            display: 'flex', alignItems: 'center',
            justifyContent: 'space-between', gap: 4,
            overflowX: 'auto',
          }}>
            {STEPS.map((step, i) => (
              <div key={step.key} style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
                <StepCard step={step} index={i} imgs={pipelineResult} activeStep={activeStep} />
                {i < STEPS.length - 1 && (
                  <span style={{ color: '#9ca3af', fontSize: '1rem', padding: '0 4px', marginBottom: 20 }}>→</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
