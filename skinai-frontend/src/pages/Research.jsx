import React from 'react';
import { FileText, Code, Database, Layers, ExternalLink } from 'lucide-react';
import DatasetAnalysis from '../components/DatasetAnalysis';
import { useReveal } from '../hooks/useReveal';

export default function Research() {
  const ref = useReveal();

  return (
    <div className="mesh-bg" style={{ minHeight: '100vh', padding: '120px 2rem 6rem', position: 'relative', overflow: 'hidden' }}>
      <div style={{ maxWidth: 1000, margin: '0 auto', position: 'relative', zIndex: 1 }}>
        
        {/* Title */}
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <h1 className="syne" style={{ fontSize: '3rem', color: '#111827', marginBottom: '1.25rem', letterSpacing: '-0.02em' }}>Research & Methodology</h1>
          <p style={{ color: '#4B5563', maxWidth: '640px', margin: '0 auto', fontSize: '1.1rem', lineHeight: '1.6' }}>
            A deep dive into the scientific foundation, data distribution, and architectural decisions powering SkinGuard.
          </p>
        </div>

        {/* Abstract */}
        <div className="reveal visible" ref={ref} style={{ background: '#fff', borderRadius: '24px', padding: '3rem', border: '1px solid #E5E7EB', marginBottom: '4rem', boxShadow: '0 4px 30px rgba(0,0,0,0.03)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '2rem' }}>
            <div style={{ background: '#2563EB', color: '#fff', padding: '12px', borderRadius: '15px' }}>
              <FileText size={28} />
            </div>
            <h2 className="syne" style={{ fontSize: '1.75rem', color: '#111827' }}>Project Abstract</h2>
          </div>
          <p style={{ color: '#4B5563', lineHeight: '1.8', fontSize: '1.1rem' }}>
            This project focuses on the automated detection and classification of skin diseases using multi-architecture deep learning ensembles. By leveraging advanced convolutional neural networks including EfficientNet-B3, InceptionV3, and ConvNeXt, we achieve high accuracy in identifying 9 distinct skin conditions. The integration of 3D soft-attention mechanisms allows the models to focus on critical dermatoscopic patterns, significantly reducing false negatives in early-stage melanoma detection.
          </p>
        </div>

        {/* Data Analysis Section */}
        <div className="reveal visible" style={{ marginBottom: '5rem' }}>
          <DatasetAnalysis />
        </div>

        {/* Key Components Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '5rem' }}>
          
          <div style={{ background: '#fff', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.02)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
              <Database size={24} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.25rem' }}>Dataset Curation</h3>
            </div>
            <p style={{ fontSize: '0.95rem', color: '#6B7280', lineHeight: '1.7' }}>
              Utilizing the ISIC 2019 archive containing over 25,000 dermatoscopic images across 8 diagnostic categories, augmented with a custom 9th "Healthy Skin" class to ensure model robustness against non-disease cases.
            </p>
          </div>

          <div style={{ background: '#fff', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', boxShadow: '0 4px 20px rgba(0,0,0,0.02)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
              <Layers size={24} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.25rem' }}>Clinical Preprocessing</h3>
            </div>
            <p style={{ fontSize: '0.95rem', color: '#6B7280', lineHeight: '1.7' }}>
              Advanced image enhancement pipeline including Gray World white balancing, BlackHat hair removal, and ITA-based CLAHE for optimal feature extraction and standardized input quality.
            </p>
          </div>

        </div>

        {/* Scientific References Section */}
        <div style={{ background: 'rgba(255,255,255,0.8)', borderRadius: '24px', padding: '3rem', border: '1px solid #E5E7EB', backdropFilter: 'blur(20px)', marginBottom: '4rem' }}>
          <h2 className="syne" style={{ fontSize: '1.75rem', marginBottom: '2.5rem', display: 'flex', alignItems: 'center', gap: '15px' }}>
            <div style={{ background: '#111827', color: '#fff', padding: '10px', borderRadius: '10px' }}>
               <Database size={24} />
            </div>
            Scientific Foundation
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {[
              { 
                title: "EFAM-Net: A Multi-Class Skin Lesion Classification Model Utilizing Enhanced Feature Fusion and Attention Mechanisms", 
                link: "https://ieeexplore.ieee.org/document/10695064", 
                year: "2024", 
                journal: "IEEE Transactions on Instrumentation and Measurement" 
              },
              { 
                title: "Multi-Class Skin Disease Classification Using Deep Learning and Transfer Learning", 
                link: "https://ieeexplore.ieee.org/document/10734113", 
                year: "2024", 
                journal: "IEEE Xplore" 
              },
              { 
                title: "Skin Cancer Classification Using Deep Learning: A Review", 
                link: "https://ieeexplore.ieee.org/document/9121248", 
                year: "2020", 
                journal: "IEEE Access" 
              },
              { 
                title: "Automatic Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 ISBI", 
                link: "https://ieeexplore.ieee.org/document/9007648", 
                year: "2018", 
                journal: "IEEE Journal of Biomedical and Health Informatics" 
              },
              { 
                title: "Multi-class skin lesion classification with deep convolutional neural networks", 
                link: "https://pmc.ncbi.nlm.nih.gov/articles/PMC11351641/", 
                year: "2024", 
                journal: "Scientific Reports" 
              }
            ].map((paper, i) => (
              <a key={i} href={paper.link} target="_blank" rel="noopener noreferrer" style={{ 
                display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                padding: '1.5rem', background: '#fff', borderRadius: '20px', 
                border: '1px solid #F3F4F6', textDecoration: 'none', transition: 'all 0.3s ease'
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-4px)'; e.currentTarget.style.borderColor = '#2563EB'; e.currentTarget.style.boxShadow = '0 10px 20px rgba(37,99,235,0.1)'; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.borderColor = '#F3F4F6'; e.currentTarget.style.boxShadow = 'none'; }}
              >
                <div>
                  <h4 style={{ color: '#111827', fontSize: '1rem', marginBottom: '6px', fontWeight: 700 }}>{paper.title}</h4>
                  <p style={{ color: '#6B7280', fontSize: '0.85rem' }}>{paper.journal} • {paper.year}</p>
                </div>
                <div style={{ color: '#2563EB' }}>
                  <ExternalLink size={20} />
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Links */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', flexWrap: 'wrap', marginTop: '4rem' }}>
          <a href="https://github.com/AnshMeshram/Skin-Disease-Detection_Project" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <button style={{ 
              display: 'flex', alignItems: 'center', gap: '10px', 
              background: '#111827', color: '#fff', border: 'none', 
              padding: '1rem 2rem', borderRadius: '999px', fontSize: '1rem', fontWeight: 600, cursor: 'pointer', transition: 'transform 0.2s' 
            }} onMouseEnter={e => e.currentTarget.style.transform = 'scale(1.05)'} onMouseLeave={e => e.currentTarget.style.transform = 'scale(1)'}>
              <Code size={20} /> View Project Source
            </button>
          </a>
        </div>

      </div>
    </div>
  );
}
