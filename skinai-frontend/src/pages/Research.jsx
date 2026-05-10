import React, { useState } from 'react';
import { 
  FileText, Code, Database, Layers, ExternalLink, Cpu, FlaskConical, BarChart3, 
  Target, Eye, GitMerge, Zap, TrendingUp, Search 
} from 'lucide-react';
import DatasetAnalysis from '../components/DatasetAnalysis';
import { useReveal } from '../hooks/useReveal';
import * as Tabs from '@radix-ui/react-tabs';

export default function Research() {
  const ref = useReveal();
  const [activeTab, setActiveTab] = useState('abstract');

  const tabTriggerStyle = (active) => ({
    padding: '0.75rem 1.5rem',
    fontSize: '0.9rem',
    fontWeight: 600,
    color: active ? '#2563EB' : '#6B7280',
    background: active ? '#fff' : 'transparent',
    border: 'none',
    borderRadius: '12px',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    boxShadow: active ? '0 4px 12px rgba(37,99,235,0.08)' : 'none',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  });

  return (
    <div className="mesh-bg" style={{ minHeight: '100vh', padding: '120px 2rem 6rem', position: 'relative', overflow: 'hidden' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        
        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', background: 'rgba(37,99,235,0.08)', padding: '6px 16px', borderRadius: '999px', color: '#2563EB', fontSize: '0.75rem', fontWeight: 800, marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            <FlaskConical size={14} /> Scientific Foundation
          </div>
          <h1 className="syne" style={{ fontSize: '3.5rem', color: '#111827', marginBottom: '1.25rem', letterSpacing: '-0.03em' }}>Technical Research</h1>
          <p style={{ color: '#4B5563', maxWidth: '640px', margin: '0 auto', fontSize: '1.1rem', lineHeight: '1.6' }}>
            A comprehensive breakdown of the architectural optimizations and scientific papers powering TwachaRakshak.
          </p>
        </div>

        {/* Radix Tabs Implementation */}
        <Tabs.Root value={activeTab} onValueChange={setActiveTab} className="reveal visible" ref={ref}>
          <Tabs.List style={{ 
            display: 'flex', 
            gap: '8px', 
            background: 'rgba(0,0,0,0.03)', 
            padding: '6px', 
            borderRadius: '16px', 
            marginBottom: '2.5rem',
            width: 'fit-content',
            margin: '0 auto 3rem'
          }}>
            <Tabs.Trigger value="abstract" style={tabTriggerStyle(activeTab === 'abstract')}>
              <FileText size={16} /> Abstract
            </Tabs.Trigger>
            <Tabs.Trigger value="dataset" style={tabTriggerStyle(activeTab === 'dataset')}>
              <Database size={16} /> Dataset
            </Tabs.Trigger>
            <Tabs.Trigger value="methodology" style={tabTriggerStyle(activeTab === 'methodology')}>
              <Cpu size={16} /> Methodology
            </Tabs.Trigger>
            <Tabs.Trigger value="papers" style={tabTriggerStyle(activeTab === 'papers')}>
              <FileText size={16} /> Scientific Papers
            </Tabs.Trigger>
          </Tabs.List>

          {/* Abstract Content */}
          <Tabs.Content value="abstract">
            <div style={{ background: '#fff', borderRadius: '24px', padding: '3.5rem', border: '1px solid #E5E7EB', boxShadow: '0 10px 40px rgba(0,0,0,0.03)', animation: 'fadeUp 0.5s ease' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '2rem' }}>
                <div style={{ background: '#2563EB', color: '#fff', padding: '12px', borderRadius: '15px' }}>
                  <FileText size={28} />
                </div>
                <h2 className="syne" style={{ fontSize: '1.75rem', color: '#111827' }}>Project Abstract</h2>
              </div>
              <p style={{ color: '#4B5563', lineHeight: '1.8', fontSize: '1.15rem' }}>
                TwachaRakshak focuses on the automated detection and classification of skin diseases using multi-architecture deep learning ensembles. By leveraging advanced convolutional neural networks including EfficientNet-B3, InceptionV3, and ConvNeXt, we achieve high accuracy in identifying 9 distinct skin conditions. The integration of 3D soft-attention mechanisms allows the models to focus on critical dermatoscopic patterns, significantly reducing false negatives in early-stage melanoma detection.
              </p>
              <div style={{ marginTop: '2.5rem', paddingTop: '2.5rem', borderTop: '1px solid #F3F4F6', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem' }}>
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
              <div style={{ marginTop: '2.5rem', background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB' }}>
                <h3 className="syne" style={{ fontSize: '1.25rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Database size={20} color="#2563EB" /> Dataset Curation
                </h3>
                <p style={{ color: '#6B7280', fontSize: '0.95rem', lineHeight: '1.7' }}>
                  Utilizing the ISIC 2019 archive containing over 25,000 dermatoscopic images across 8 diagnostic categories, augmented with a custom 9th "Healthy Skin" class. The data underwent rigorous cleaning, duplicate removal, and class-balancing using Focal Loss weighting.
                </p>
              </div>
            </div>
          </Tabs.Content>

          {/* Methodology Content */}
          <Tabs.Content value="methodology">
            <div style={{ animation: 'fadeUp 0.5s ease' }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                {[
                  {
                    title: "Label Smoothing Focal Loss",
                    desc: "Combats class imbalance by dynamically weighting 'hard' samples and smoothing labels to prevent overconfident clinical misclassifications.",
                    icon: <Target size={24} color="#EF4444" />, 
                    bg: "#FEF2F2"
                  },
                  {
                    title: "3D Soft Attention Unit",
                    desc: "A custom spatial weighting layer that focuses on multi-scale texture features while suppressing noise from skin hair and imaging artifacts.",
                    icon: <Eye size={24} color="#8B5CF6" />,
                    bg: "#F5F3FF"
                  },
                  {
                    title: "Weighted Ensemble Voting",
                    desc: "Combines EfficientNet-B3, InceptionV3, and ConvNeXt-Tiny predictions using a learned weight matrix for maximum diagnostic consensus.",
                    icon: <GitMerge size={24} color="#F59E0B" />,
                    bg: "#FFFBEB"
                  },
                  {
                    title: "Test-Time Augmentation (TTA)",
                    desc: "Runs multiple transformations (rotations/flips) on a single input during inference to ensure stability against varying camera angles.",
                    icon: <Zap size={24} color="#10B981" />,
                    bg: "#ECFDF5"
                  },
                  {
                    title: "Cosine Annealing",
                    desc: "Utilizes cyclical learning rates with warm restarts to escape local minima, ensuring the model reaches a superior global optimum.",
                    icon: <TrendingUp size={24} color="#3B82F6" />,
                    bg: "#EFF6FF"
                  },
                  {
                    title: "Grad-CAM Explainability",
                    desc: "Generates visual heatmaps highlighting the exact pixel regions used for prediction, providing clinical transparency for medical users.",
                    icon: <Search size={24} color="#EC4899" />,
                    bg: "#FDF2F8"
                  },
                  {
                    title: "Compound Scaling",
                    desc: "Balances depth, width, and resolution using the EfficientNet architecture for optimal feature extraction from dermatoscopic imagery.",
                    icon: <Layers size={24} color="#6366F1" />,
                    bg: "#EEF2FF"
                  },
                  {
                    title: "Mixed Precision FP16",
                    desc: "Optimized for high-speed inference on modern GPU architectures without sacrificing diagnostic sensitivity or specificity.",
                    icon: <Cpu size={24} color="#14B8A6" />,
                    bg: "#F0FDFA"
                  }
                ].map((tech, i) => (
                  <div key={i} style={{ 
                    background: '#fff', padding: '2rem', borderRadius: '24px', border: '1px solid #E5E7EB', 
                    boxShadow: '0 4px 20px rgba(0,0,0,0.02)', transition: 'all 0.3s ease',
                    display: 'flex', flexDirection: 'column', gap: '1rem'
                  }}
                  onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-5px)'; e.currentTarget.style.borderColor = '#2563EB'; }}
                  onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.borderColor = '#E5E7EB'; }}
                  >
                    <div style={{ 
                      width: '48px', height: '48px', borderRadius: '14px', 
                      background: tech.bg, display: 'flex', alignItems: 'center', 
                      justifyContent: 'center'
                    }}>
                      {tech.icon}
                    </div>
                    <div>
                      <h4 className="syne" style={{ fontSize: '1.1rem', color: '#111827', marginBottom: '0.75rem', fontWeight: 700 }}>{tech.title}</h4>
                      <p style={{ fontSize: '0.88rem', color: '#6B7280', lineHeight: '1.6' }}>{tech.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Tabs.Content>

          {/* Papers Content */}
          <Tabs.Content value="papers">
            <div style={{ background: '#fff', borderRadius: '24px', padding: '3rem', border: '1px solid #E5E7EB', boxShadow: '0 4px 30px rgba(0,0,0,0.03)', animation: 'fadeUp 0.5s ease' }}>
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
                      <h4 style={{ color: '#111827', fontSize: '0.95rem', marginBottom: '4px', fontWeight: 700 }}>{paper.title}</h4>
                      <p style={{ color: '#6B7280', fontSize: '0.8rem' }}>{paper.journal} • {paper.year}</p>
                    </div>
                    <ExternalLink size={18} color="#2563EB" />
                  </a>
                ))}
              </div>
            </div>
          </Tabs.Content>
        </Tabs.Root>

        {/* Bottom Actions */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', marginTop: '5rem' }}>
          <a href="https://github.com/AnshMeshram/Skin-Disease-Detection_Project" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <button style={{ 
              display: 'flex', alignItems: 'center', gap: '10px', 
              background: '#111827', color: '#fff', border: 'none', 
              padding: '1.25rem 2.5rem', borderRadius: '999px', fontSize: '1rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s' 
            }} onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'} onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}>
              <Code size={22} /> Explore Neural Source Code
            </button>
          </a>
        </div>

      </div>
    </div>
  );
}
