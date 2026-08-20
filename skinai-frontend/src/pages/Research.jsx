import React, { useState } from 'react';
import { 
  FileText, Code, Database, Layers, ExternalLink, Cpu, FlaskConical, BarChart3, 
  Target, Eye, GitMerge, Zap, TrendingUp, Search, Sparkles, Activity, ShieldCheck 
} from 'lucide-react';
import DatasetAnalysis from '../components/DatasetAnalysis';
import { useReveal } from '../hooks/useReveal';
import * as Tabs from '@radix-ui/react-tabs';

export default function Research() {
  const ref = useReveal();
  const [activeTab, setActiveTab] = useState('abstract');

  // Support direct URL hash linking (e.g. /research#dataset)
  React.useEffect(() => {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    if (['abstract', 'dataset', 'methodology', 'papers'].includes(hash)) {
      setActiveTab(hash);
    }
  }, []);

  const tabTriggerStyle = (active) => ({
    padding: '10px 24px',
    fontSize: '0.9375rem',
    fontWeight: 700,
    color: active ? '#fff' : '#4B5563',
    background: active ? '#2563EB' : 'rgba(255, 255, 255, 0.8)',
    border: active ? '1px solid #2563EB' : '1px solid rgba(0,0,0,0.06)',
    borderRadius: '12px',
    cursor: 'pointer',
    transition: 'all 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
    boxShadow: active ? '0 8px 20px rgba(37,99,235,0.25)' : '0 2px 8px rgba(0,0,0,0.04)',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  return (
    <div className="mesh-bg" style={{ minHeight: '100vh', padding: '120px 2rem 6rem', position: 'relative', overflow: 'hidden' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        
        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(37,99,235,0.08)', padding: '6px 16px', borderRadius: '999px', color: '#2563EB', fontSize: '0.75rem', fontWeight: 800, marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            <FlaskConical size={14} /> Scientific Foundation
          </div>
          <h1 className="syne" style={{ fontSize: 'clamp(2.25rem, 5vw, 3rem)', color: '#111827', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>Technical Research</h1>
          <p style={{ color: '#4B5563', maxWidth: '640px', margin: '0 auto', fontSize: '0.9375rem', lineHeight: '1.65' }}>
            A comprehensive breakdown of the architectural optimizations, ISIC 2019 dataset distributions, and scientific papers powering TwachaRakshak.
          </p>
        </div>

        {/* Radix Tabs Implementation */}
        <Tabs.Root value={activeTab} onValueChange={(val) => { setActiveTab(val); window.location.hash = val; }} className="reveal visible" ref={ref}>
          <Tabs.List className="sticky-research-tabs" style={{ 
            display: 'flex', 
            justifyContent: 'center',
            gap: '10px', 
            background: 'rgba(255, 255, 255, 0.65)', 
            backdropFilter: 'blur(20px)',
            padding: '8px', 
            borderRadius: '20px', 
            border: '1px solid rgba(255, 255, 255, 0.8)',
            boxShadow: '0 8px 30px rgba(0, 0, 0, 0.04)',
            marginBottom: '3rem',
            width: 'fit-content',
            margin: '0 auto 3rem',
            position: 'sticky',
            top: '85px',
            zIndex: 50,
            maxWidth: '100%'
          }}>
            <Tabs.Trigger value="abstract" style={tabTriggerStyle(activeTab === 'abstract')}>
              <FileText size={18} /> Abstract
            </Tabs.Trigger>
            <Tabs.Trigger value="dataset" style={tabTriggerStyle(activeTab === 'dataset')}>
              <Database size={18} /> Dataset & Charts
            </Tabs.Trigger>
            <Tabs.Trigger value="methodology" style={tabTriggerStyle(activeTab === 'methodology')}>
              <Cpu size={18} /> Methodology
            </Tabs.Trigger>
            <Tabs.Trigger value="papers" style={tabTriggerStyle(activeTab === 'papers')}>
              <FileText size={18} /> Publications
            </Tabs.Trigger>
          </Tabs.List>

          {/* Abstract Content */}
          <Tabs.Content value="abstract">
            <div style={{ background: '#fff', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', animation: 'fadeUp 0.5s ease' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '2rem' }}>
                <div style={{ background: '#2563EB', color: '#fff', padding: '12px', borderRadius: '15px' }}>
                  <FileText size={28} />
                </div>
                <h2 className="syne" style={{ fontSize: '1.75rem', color: '#111827' }}>Project Abstract</h2>
              </div>
              <p style={{ color: '#4B5563', lineHeight: '1.65', fontSize: '0.9375rem' }}>
                TwachaRakshak focuses on the automated detection and classification of skin diseases using multi-architecture deep learning ensembles. By leveraging advanced convolutional neural networks including EfficientNet-B3, InceptionV3, and ConvNeXt, we achieve high accuracy in identifying 9 distinct skin conditions. The integration of 3D soft-attention mechanisms allows the models to focus on critical dermatoscopic patterns, significantly reducing false negatives in early-stage melanoma detection.
              </p>
              <div style={{ marginTop: '2.5rem', paddingTop: '2.5rem', borderTop: '1px solid #F3F4F6', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '1.5rem' }}>
                <div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#2563EB', marginBottom: '4px' }}>95.48%</div>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 600, textTransform: 'uppercase' }}>Balanced Accuracy</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#10B981', marginBottom: '4px' }}>9 Classes</div>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 600, textTransform: 'uppercase' }}>Condition Labels</div>
                </div>
                <div>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#8B5CF6', marginBottom: '4px' }}>3-Model</div>
                  <div style={{ fontSize: '0.75rem', color: '#6B7280', fontWeight: 600, textTransform: 'uppercase' }}>Weighted Ensemble</div>
                </div>
              </div>
            </div>
          </Tabs.Content>

          {/* Dataset Content */}
          <Tabs.Content value="dataset">
            <div style={{ animation: 'fadeUp 0.5s ease' }}>
              <DatasetAnalysis />
              <div style={{ marginTop: '2.5rem', background: '#fff', padding: '2rem', borderRadius: '20px', border: '1px solid #E5E7EB' }}>
                <h3 className="syne" style={{ fontSize: '1.25rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Database size={20} color="#2563EB" /> Dataset Curation
                </h3>
                <p style={{ color: '#6B7280', fontSize: '0.8125rem', lineHeight: '1.65' }}>
                  Utilizing the ISIC 2019 archive containing over 25,000 dermatoscopic images across 8 diagnostic categories, augmented with a custom 9th "Healthy Skin" class. The data underwent rigorous cleaning, duplicate removal, and class-balancing using Focal Loss weighting.
                </p>
              </div>
            </div>
          </Tabs.Content>

          {/* Methodology Content */}
          <Tabs.Content value="methodology">
            <div style={{ animation: 'fadeUp 0.5s ease', display: 'flex', flexDirection: 'column', gap: '2.5rem' }}>
              
              {/* Section 1: Preprocessing & Standardization */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
                  <div style={{ background: '#EFF6FF', color: '#2563EB', padding: '8px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 800 }}>
                    PIPELINE 01
                  </div>
                  <h3 className="syne" style={{ fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                    Dermoscopic Image Preprocessing &amp; Standardization
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  {[
                    {
                      title: "Adaptive DullRazor Hair Removal v3.4",
                      desc: "Directional linear morphological closing with 15° kernel increments erases thick hair shafts without eroding micro-vascular structures or pigment networks.",
                      icon: <Layers size={22} color="#2563EB" />,
                      bg: "#EFF6FF"
                    },
                    {
                      title: "Shades-of-Grey Color Constancy",
                      desc: "Normalizes lighting variations across different dermatoscope brands (Heine, DermLite) using Minkowski L6-norm illumination vector estimation.",
                      icon: <Sparkles size={22} color="#10B981" />,
                      bg: "#ECFDF5"
                    },
                    {
                      title: "LAB-Space CLAHE Contrast Tuning",
                      desc: "Applies Contrast Limited Adaptive Histogram Equalization to the L* lightness channel to heighten subtle lesion borders without altering native skin pigmentation.",
                      icon: <Activity size={22} color="#8B5CF6" />,
                      bg: "#F5F3FF"
                    }
                  ].map((item, i) => (
                    <div key={i} style={{
                      background: '#fff', padding: '1.75rem', borderRadius: '20px', border: '1px solid #E5E7EB',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
                    }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: item.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {item.icon}
                      </div>
                      <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0, fontWeight: 700 }}>{item.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.6', margin: 0 }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 2: Class Imbalance & Loss Function Optimization */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
                  <div style={{ background: '#FEF2F2', color: '#DC2626', padding: '8px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 800 }}>
                    PIPELINE 02
                  </div>
                  <h3 className="syne" style={{ fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                    Class Imbalance &amp; Loss Function Optimization
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  {[
                    {
                      title: "Label Smoothing Focal Loss",
                      desc: "Combats extreme dataset imbalance by dynamically downweighting easy negative samples (γ=2.0, α=0.25) while smoothing soft targets (ε=0.1) to prevent overconfident boundary errors.",
                      icon: <Target size={22} color="#DC2626" />,
                      bg: "#FEF2F2"
                    },
                    {
                      title: "Asymmetric Cost-Sensitive Weighting",
                      desc: "Imposes an 8× higher loss penalty on false negatives for malignant tumors (Melanoma, BCC, SCC) compared to benign growths, prioritizing clinical safety.",
                      icon: <ShieldCheck size={22} color="#D97706" />,
                      bg: "#FFFBEB"
                    },
                    {
                      title: "Stratified 5-Fold Cross Validation",
                      desc: "Guarantees proportional diagnostic class ratios across all 5 training, validation, and holdout test splits for unbiased cross-fold evaluation.",
                      icon: <BarChart3 size={22} color="#2563EB" />,
                      bg: "#EFF6FF"
                    }
                  ].map((item, i) => (
                    <div key={i} style={{
                      background: '#fff', padding: '1.75rem', borderRadius: '20px', border: '1px solid #E5E7EB',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
                    }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: item.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {item.icon}
                      </div>
                      <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0, fontWeight: 700 }}>{item.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.6', margin: 0 }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 3: Model Architecture & Attention Mechanism */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
                  <div style={{ background: '#F5F3FF', color: '#7C3AED', padding: '8px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 800 }}>
                    PIPELINE 03
                  </div>
                  <h3 className="syne" style={{ fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                    Model Architecture &amp; 3D Soft-Attention Units
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  {[
                    {
                      title: "3D Soft-Attention Mechanism",
                      desc: "Custom spatial-channel weighting modules inside EfficientNet and ConvNeXt backbones focus neural activations on central lesion morphology while suppressing peripheral noise.",
                      icon: <Eye size={22} color="#7C3AED" />,
                      bg: "#F5F3FF"
                    },
                    {
                      title: "Multi-Scale Feature Pyramid Fusion",
                      desc: "Fuses high-resolution low-level texture maps with deep semantic feature vectors for simultaneous detection of micro-dots and macroscopic lesion asymmetry.",
                      icon: <GitMerge size={22} color="#D97706" />,
                      bg: "#FFFBEB"
                    },
                    {
                      title: "Compound Scaling & Depthwise Convolutions",
                      desc: "Balances network depth, width, and input resolution (300×300) with 7×7 depthwise convolutions for maximal parameter efficiency.",
                      icon: <Cpu size={22} color="#2563EB" />,
                      bg: "#EFF6FF"
                    }
                  ].map((item, i) => (
                    <div key={i} style={{
                      background: '#fff', padding: '1.75rem', borderRadius: '20px', border: '1px solid #E5E7EB',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
                    }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: item.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {item.icon}
                      </div>
                      <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0, fontWeight: 700 }}>{item.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.6', margin: 0 }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 4: Ensemble & Inference Engineering */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
                  <div style={{ background: '#FFFBEB', color: '#D97706', padding: '8px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 800 }}>
                    PIPELINE 04
                  </div>
                  <h3 className="syne" style={{ fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                    Ensemble &amp; Inference Optimization Engineering
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  {[
                    {
                      title: "Learned Temperature-Scaled Consensus",
                      desc: "Calibrates model logits with architecture-specific temperature parameters (T ∈ [1.2, 1.8]) prior to Softmax, preventing overconfident single-model dominance.",
                      icon: <TrendingUp size={22} color="#D97706" />,
                      bg: "#FFFBEB"
                    },
                    {
                      title: "Mixed-Precision FP16 CUDA Acceleration",
                      desc: "Utilizes PyTorch Automatic Mixed Precision (autocast) to cut GPU VRAM footprint by 50% while accelerating inference latency to < 45ms per scan.",
                      icon: <Cpu size={22} color="#2563EB" />,
                      bg: "#EFF6FF"
                    },
                    {
                      title: "ONNX Runtime & TensorRT Export Ready",
                      desc: "Pre-configured computation graphs for export to ONNX and NVIDIA TensorRT execution providers for zero-latency local edge deployment.",
                      icon: <Code size={22} color="#059669" />,
                      bg: "#ECFDF5"
                    }
                  ].map((item, i) => (
                    <div key={i} style={{
                      background: '#fff', padding: '1.75rem', borderRadius: '20px', border: '1px solid #E5E7EB',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
                    }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: item.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {item.icon}
                      </div>
                      <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0, fontWeight: 700 }}>{item.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.6', margin: 0 }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Section 5: Clinical Explainability & Safety Guards */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1.25rem' }}>
                  <div style={{ background: '#ECFDF5', color: '#059669', padding: '8px 12px', borderRadius: '8px', fontSize: '0.75rem', fontWeight: 800 }}>
                    PIPELINE 05
                  </div>
                  <h3 className="syne" style={{ fontSize: '1.25rem', color: '#111827', margin: 0 }}>
                    Clinical Explainability &amp; Safety Verification
                  </h3>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  {[
                    {
                      title: "High-Fidelity Grad-CAM Heatmaps",
                      desc: "Generates pixel-level spatial attribution maps overlaying raw dermoscopy scans to visually confirm model alignment with dermatological ABCD criteria.",
                      icon: <Search size={22} color="#059669" />,
                      bg: "#ECFDF5"
                    },
                    {
                      title: "Monte Carlo Dropout Uncertainty Lock",
                      desc: "Executes N=10 stochastic forward passes at test time to quantify predictive variance; ambiguous scans automatically trigger an 'Inconclusive' warning.",
                      icon: <Target size={22} color="#DC2626" />,
                      bg: "#FEF2F2"
                    },
                    {
                      title: "5-View Test-Time Augmentation (TTA)",
                      desc: "Evaluates 5 augmented rotations and flips during inference to guarantee orientation-invariant predictions across mobile devices.",
                      icon: <Zap size={22} color="#2563EB" />,
                      bg: "#EFF6FF"
                    }
                  ].map((item, i) => (
                    <div key={i} style={{
                      background: '#fff', padding: '1.75rem', borderRadius: '20px', border: '1px solid #E5E7EB',
                      boxShadow: '0 4px 20px rgba(0,0,0,0.02)', display: 'flex', flexDirection: 'column', gap: '0.75rem'
                    }}>
                      <div style={{ width: '44px', height: '44px', borderRadius: '12px', background: item.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        {item.icon}
                      </div>
                      <h4 className="syne" style={{ fontSize: '1.05rem', color: '#0F172A', margin: 0, fontWeight: 700 }}>{item.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: '#64748B', lineHeight: '1.6', margin: 0 }}>{item.desc}</p>
                    </div>
                  ))}
                </div>
              </div>

            </div>
          </Tabs.Content>

          {/* Papers Content */}
          <Tabs.Content value="papers">
            <div style={{ background: '#fff', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.04)', animation: 'fadeUp 0.5s ease' }}>
              <h2 className="syne" style={{ fontSize: '1.75rem', marginBottom: '2.5rem', display: 'flex', alignItems: 'center', gap: '15px' }}>
                <FileText size={24} color="#2563EB" /> Academic References
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                {[
                  { title: "EFAM-Net: Multi-Class Skin Lesion Classification", link: "https://ieeexplore.ieee.org/document/10695064", year: "2024", journal: "IEEE TIM" },
                  { title: "Multi-Class Skin Disease Classification", link: "https://ieeexplore.ieee.org/document/10734113", year: "2024", journal: "IEEE Xplore" },
                  { title: "Skin Cancer Classification Review", link: "https://ieeexplore.ieee.org/document/9121248", year: "2020", journal: "IEEE Access" },
                  { title: "Melanoma Detection Challenge ISBI", link: "https://ieeexplore.ieee.org/document/9007648", year: "2018", journal: "IEEE JBHI" }
                ].map((paper, i) => (
                  <a key={i} href={paper.link} target="_blank" rel="noopener noreferrer" style={{ 
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                    padding: '1.5rem', background: '#F9FAFB', borderRadius: '16px', 
                    border: '1px solid #F3F4F6', textDecoration: 'none', transition: 'all 0.2s ease'
                  }}
                  onMouseEnter={e => { e.currentTarget.style.background = '#fff'; e.currentTarget.style.borderColor = '#2563EB'; }}
                  onMouseLeave={e => { e.currentTarget.style.background = '#F9FAFB'; e.currentTarget.style.borderColor = '#F3F4F6'; }}
                  >
                    <div>
                      <h4 style={{ color: '#111827', fontSize: '0.9375rem', marginBottom: '4px', fontWeight: 700 }}>{paper.title}</h4>
                      <p style={{ color: '#6B7280', fontSize: '0.8125rem' }}>{paper.journal} • {paper.year}</p>
                    </div>
                    <ExternalLink size={18} color="#2563EB" />
                  </a>
                ))}
              </div>
            </div>
          </Tabs.Content>
        </Tabs.Root>

        {/* Bottom Actions */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', marginTop: '3rem' }}>
          <a href="https://github.com/AnshMeshram/Skin-Disease-Detection_Project" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <button style={{ 
              display: 'flex', alignItems: 'center', gap: '10px', 
              background: '#111827', color: '#fff', border: 'none', 
              padding: '14px 32px', borderRadius: '999px', fontSize: '0.9375rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s' 
            }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
              <Code size={22} /> Explore Neural Source Code
            </button>
          </a>
        </div>

      </div>
    </div>
  );
}
