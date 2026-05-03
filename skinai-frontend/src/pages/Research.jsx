import React from 'react';
import { FileText, Code, Database, Layers, ExternalLink } from 'lucide-react';

export default function Research() {
  return (
    <div style={{ background: '#ECECEC', minHeight: '100vh', padding: '100px 2rem 4rem' }}>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        
        {/* Title */}
        <div style={{ marginBottom: '3rem' }}>
          <h1 className="syne" style={{ fontSize: '2.5rem', color: '#111827', marginBottom: '1rem' }}>Research & Methodology</h1>
          <div style={{ width: '60px', height: '4px', background: '#2563EB', borderRadius: '2px' }}></div>
        </div>

        {/* Abstract */}
        <div style={{ background: '#fff', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', marginBottom: '2.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
            <FileText size={24} color="#2563EB" />
            <h2 className="syne" style={{ fontSize: '1.5rem' }}>Project Abstract</h2>
          </div>
          <p style={{ color: '#4B5563', lineHeight: '1.8', fontSize: '1rem' }}>
            This project focuses on the automated detection and classification of skin diseases using multi-architecture deep learning ensembles. By leveraging advanced convolutional neural networks including EfficientNet-B3, InceptionV3, and ConvNeXt, we achieve high accuracy in identifying 9 distinct skin conditions. The integration of 3D soft-attention mechanisms allows the models to focus on critical dermatoscopic patterns, significantly reducing false negatives in early-stage melanoma detection.
          </p>
        </div>

        {/* Key Components Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '3rem' }}>
          
          <div style={{ background: '#fff', borderRadius: '20px', padding: '2rem', border: '1px solid #E5E7EB' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
              <Database size={20} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.1rem' }}>Dataset</h3>
            </div>
            <p style={{ fontSize: '0.85rem', color: '#6B7280', lineHeight: '1.6' }}>
              Utilizing the ISIC 2019 archive containing over 25,000 dermatoscopic images across 8 diagnostic categories, augmented with a custom 9th "Healthy Skin" class.
            </p>
          </div>

          <div style={{ background: '#fff', borderRadius: '20px', padding: '2rem', border: '1px solid #E5E7EB' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
              <Layers size={20} color="#2563EB" />
              <h3 className="syne" style={{ fontSize: '1.1rem' }}>Preprocessing</h3>
            </div>
            <p style={{ fontSize: '0.85rem', color: '#6B7280', lineHeight: '1.6' }}>
              Advanced image enhancement pipeline including Gray World white balancing, BlackHat hair removal, and ITA-based CLAHE for optimal feature extraction.
            </p>
          </div>

        </div>

        {/* Links */}
        {/* Links */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '4rem' }}>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <button style={{ 
              display: 'flex', alignItems: 'center', gap: '8px', 
              background: '#111827', color: '#fff', border: 'none', 
              padding: '0.75rem 1.5rem', borderRadius: '999px', fontSize: '0.9rem', cursor: 'pointer' 
            }}>
              <Code size={18} /> View Source Code
            </button>
          </a>
          <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer" style={{ textDecoration: 'none' }}>
            <button style={{ 
              display: 'flex', alignItems: 'center', gap: '8px', 
              background: '#fff', color: '#111827', border: '1px solid #E5E7EB', 
              padding: '0.75rem 1.5rem', borderRadius: '999px', fontSize: '0.9rem', cursor: 'pointer' 
            }}>
              <FileText size={18} /> Research Document <ExternalLink size={14} />
            </button>
          </a>
        </div>

        {/* Scientific References Section */}
        <div style={{ background: 'rgba(255,255,255,0.6)', borderRadius: '24px', padding: '2.5rem', border: '1px solid #E5E7EB', backdropFilter: 'blur(10px)' }}>
          <h2 className="syne" style={{ fontSize: '1.5rem', marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Database size={24} color="#2563EB" />
            Scientific References
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {[
              { title: "ISIC 2019: Analysis of Skin Lesions", link: "https://arxiv.org/abs/1902.03368", year: "2019", journal: "arXiv Pre-print" },
              { title: "EfficientNet: Rethinking Model Scaling for CNNs", link: "https://arxiv.org/abs/1905.11946", year: "2020", journal: "ICML" },
              { title: "Attention Is All You Need", link: "https://arxiv.org/abs/1706.03762", year: "2017", journal: "NeurIPS" },
              { title: "ConvNeXt: A ConvNet for the 2020s", link: "https://arxiv.org/abs/2201.03545", year: "2022", journal: "CVPR" },
              { title: "Deep Learning for Melanoma Recognition", link: "https://www.nature.com/articles/nature21056", year: "2017", journal: "Nature" }
            ].map((paper, i) => (
              <a key={i} href={paper.link} target="_blank" rel="noopener noreferrer" style={{ 
                display: 'flex', justifyContent: 'space-between', alignItems: 'center', 
                padding: '1.25rem', background: '#fff', borderRadius: '16px', 
                border: '1px solid #F3F4F6', textDecoration: 'none', transition: 'all 0.2s'
              }}
              onMouseEnter={e => { e.currentTarget.style.transform = 'translateX(8px)'; e.currentTarget.style.borderColor = '#2563EB'; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'translateX(0)'; e.currentTarget.style.borderColor = '#F3F4F6'; }}
              >
                <div>
                  <h4 style={{ color: '#111827', fontSize: '0.95rem', marginBottom: '4px' }}>{paper.title}</h4>
                  <p style={{ color: '#6B7280', fontSize: '0.8rem' }}>{paper.journal} • {paper.year}</p>
                </div>
                <ExternalLink size={16} color="#9CA3AF" />
              </a>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
