import React from 'react';
import ModelComparisonTable from '../components/ModelComparisonTable';
import { useReveal } from '../hooks/useReveal';
import { BarChart3, TrendingUp, PieChart, Info, Layers, Target } from 'lucide-react';

const CLASS_DIST = [
  { name: 'Melanoma', val: 12, color: '#EF4444' },
  { name: 'Melanocytic Nevus', val: 35, color: '#10B981' },
  { name: 'Basal Cell Carcinoma', val: 10, color: '#F59E0B' },
  { name: 'Actinic Keratosis', val: 8, color: '#3B82F6' },
  { name: 'Benign Keratosis', val: 11, color: '#6366F1' },
  { name: 'Dermatofibroma', val: 6, color: '#8B5CF6' },
  { name: 'Vascular Lesion', val: 5, color: '#EC4899' },
  { name: 'Squamous Cell Carcinoma', val: 8, color: '#B91C1C' },
  { name: 'Healthy Skin', val: 5, color: '#059669' }
];

export default function Model() {
  const ref = useReveal();

  return (
    <div style={{ background: '#ECECEC', minHeight: '100vh', padding: '120px 2rem 4rem' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: '#DBEAFE', color: '#1E40AF', padding: '6px 16px', borderRadius: '999px', fontSize: '0.75rem', fontWeight: 700, marginBottom: '1rem' }}>
             <Target size={14} /> BENCHMARK v2.1
          </div>
          <h1 className="syne" style={{ fontSize: '3rem', color: '#111827', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>Architecture Performance</h1>
          <p style={{ color: '#4B5563', maxWidth: '640px', margin: '0 auto', fontSize: '1.1rem', lineHeight: '1.6' }}>
            A comparative analysis of our ensemble members. We leverage the unique inductive biases of CNNs and Transformer-inspired designs.
          </p>
        </div>

        {/* Comparison Table Section */}
        <div className="reveal visible" ref={ref} style={{ marginBottom: '5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '2rem' }}>
            <div style={{ background: '#2563EB', color: '#fff', padding: '10px', borderRadius: '12px' }}>
               <TrendingUp size={24} />
            </div>
            <h2 className="syne" style={{ fontSize: '1.75rem', color: '#111827' }}>Model Benchmark Matrix</h2>
          </div>
          <ModelComparisonTable />
        </div>

        {/* Graphs Section */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: '2.5rem' }}>
          
          {/* Training Progress Graph */}
          <div style={{ background: '#fff', padding: '2.5rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '2rem' }}>
              <BarChart3 size={22} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.25rem' }}>Accuracy Convergence</h3>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
               {[
                 { label: 'EfficientNet-B3 (Validation)', val: '81.6%', width: '81.6%', color: '#10B981' },
                 { label: 'InceptionV3 (Validation)', val: '78.8%', width: '78.8%', color: '#3B82F6' },
                 { label: 'ConvNeXt Tiny (Validation)', val: '82.1%', width: '82.1%', color: '#A855F7' },
                 { label: 'Ensemble Majority (Test)', val: '84.4%', width: '84.4%', color: '#111827' }
               ].map((item, i) => (
                 <div key={i}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '8px', color: '#4B5563' }}>
                       <span style={{ fontWeight: 600 }}>{item.label}</span>
                       <span style={{ fontWeight: 700, color: item.color }}>{item.val}</span>
                    </div>
                    <div style={{ height: '8px', background: '#F3F4F6', borderRadius: '4px', overflow: 'hidden' }}>
                       <div style={{ height: '100%', width: item.width, background: item.color, borderRadius: '4px', animation: 'barGrow 1.5s ease-out' }} />
                    </div>
                 </div>
               ))}
            </div>
            
            <p style={{ marginTop: '2rem', fontSize: '0.8rem', color: '#9CA3AF', textAlign: 'center', fontStyle: 'italic' }}>
              * Metrics based on 5-fold cross-validation on ISIC 2019 dataset
            </p>
          </div>

          {/* Class Distribution Pie-style Chart */}
          <div style={{ background: '#fff', padding: '2.5rem', borderRadius: '24px', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.03)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '2rem' }}>
              <PieChart size={22} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.25rem' }}>Dataset Distribution</h3>
            </div>
            
            <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center' }}>
               {/* Visual representation of a ring chart */}
               <div style={{ position: 'relative', width: '160px', height: '160px', flexShrink: 0 }}>
                  <svg viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="18" cy="18" r="15.9" fill="transparent" stroke="#F3F4F6" strokeWidth="3.5" />
                    <circle cx="18" cy="18" r="15.9" fill="transparent" stroke="#10B981" strokeWidth="3.5" strokeDasharray="42 100" />
                    <circle cx="18" cy="18" r="15.9" fill="transparent" stroke="#EF4444" strokeWidth="3.5" strokeDasharray="18 100" strokeDashoffset="-42" />
                    <circle cx="18" cy="18" r="15.9" fill="transparent" stroke="#3B82F6" strokeWidth="3.5" strokeDasharray="40 100" strokeDashoffset="-60" />
                  </svg>
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column' }}>
                     <span className="syne" style={{ fontSize: '1.2rem', fontWeight: 800, color: '#111827' }}>25K+</span>
                     <span style={{ fontSize: '0.6rem', color: '#9CA3AF', fontWeight: 700 }}>IMAGES</span>
                  </div>
               </div>

               <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px', flex: 1 }}>
                  {CLASS_DIST.map((c, i) => (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                       <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: c.color }} />
                       <span style={{ fontSize: '0.75rem', color: '#4B5563', flex: 1 }}>{c.name}</span>
                       <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#111827' }}>{c.val}%</span>
                    </div>
                  ))}
               </div>
            </div>

            <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', gap: '10px', background: '#F9FAFB', padding: '1rem', borderRadius: '12px' }}>
               <Layers size={16} color="#6B7280" />
               <p style={{ fontSize: '0.75rem', color: '#6B7280' }}>
                 Dataset augmented using rotation, scaling, and flip transformations to improve ensemble robustness.
               </p>
            </div>
          </div>

        </div>

        {/* Methodology Card */}
        <div style={{ marginTop: '4rem', background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)', borderRadius: '24px', padding: '3rem', color: '#fff', display: 'flex', gap: '2.5rem', alignItems: 'center', boxShadow: '0 20px 25px -5px rgba(37, 99, 235, 0.2)' }}>
          <div style={{ background: 'rgba(255,255,255,0.15)', padding: '1.5rem', borderRadius: '20px', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.2)' }}>
             <Info size={40} />
          </div>
          <div>
            <h3 className="syne" style={{ fontSize: '1.5rem', marginBottom: '0.75rem', fontWeight: 700 }}>Training Protocol</h3>
            <p style={{ fontSize: '1rem', opacity: 0.9, lineHeight: '1.7' }}>
              We employ a synchronized training loop across all ensemble members. Each model is initialized with ImageNet weights and fine-tuned using a tiered learning rate strategy. The final prediction is a weighted average of individual model logits, optimized via a validation-set grid search.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
