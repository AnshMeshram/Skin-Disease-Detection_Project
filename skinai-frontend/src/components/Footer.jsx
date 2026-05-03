import React from 'react';
import { Link } from 'react-router-dom';
import { Mail, Code, Globe, ShieldCheck } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{ background: '#0F172A', padding: '4rem 2rem 2rem', color: '#fff' }}>
      <div style={{ maxWidth: 1100, margin: '0 auto' }}>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '3rem', marginBottom: '3rem' }}>
          
          {/* Brand */}
          <div style={{ gridColumn: 'span 2' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1.5rem' }}>
              <div style={{
                width: '32px', height: '32px', borderRadius: '8px',
                background: 'linear-gradient(135deg, #10B981 0%, #059669 100%)',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              </div>
              <div style={{ fontFamily: 'Outfit, sans-serif', fontSize: '1.5rem', fontWeight: 800 }}>
                Twacha<span style={{ color: '#10B981' }}>Rakshak</span>
              </div>
            </div>
            <p style={{ fontSize: '0.9rem', color: '#E2E8F0', lineHeight: '1.7', maxWidth: '340px' }}>
              Advanced dermatological AI diagnostic support system using multi-model ensemble architectures and 3D-CA attention mechanisms.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="syne" style={{ fontSize: '1rem', marginBottom: '1.25rem', color: '#F8FAFC' }}>Navigation</h4>
            <ul style={{ listStyle: 'none', padding: 0, display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <li><Link to="/" style={{ color: '#E2E8F0', fontSize: '0.85rem', textDecoration: 'none' }}>Home / Detection</Link></li>
              <li><Link to="/guide" style={{ color: '#E2E8F0', fontSize: '0.85rem', textDecoration: 'none' }}>Skin Disease Guide</Link></li>
              <li><Link to="/model" style={{ color: '#E2E8F0', fontSize: '0.85rem', textDecoration: 'none' }}>Model Benchmarks</Link></li>
              <li><Link to="/research" style={{ color: '#E2E8F0', fontSize: '0.85rem', textDecoration: 'none' }}>Research Paper</Link></li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="syne" style={{ fontSize: '1rem', marginBottom: '1.25rem', color: '#F8FAFC' }}>Connect</h4>
            <div style={{ display: 'flex', gap: '12px' }}>
               <a href="#" style={{ color: '#E2E8F0' }}><Code size={20} /></a>
               <a href="#" style={{ color: '#E2E8F0' }}><Mail size={20} /></a>
               <a href="#" style={{ color: '#E2E8F0' }}><Globe size={20} /></a>
            </div>
          </div>

        </div>

        {/* Disclaimer Bar */}
        <div style={{ 
          borderTop: '1px solid rgba(255,255,255,0.05)', 
          paddingTop: '2rem', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.5rem'
        }}>
           <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#94A3B8', fontSize: '0.75rem' }}>
              <ShieldCheck size={14} />
              <span>For research and educational purposes only. Not a substitute for clinical diagnosis.</span>
           </div>
           <p style={{ color: '#94A3B8', fontSize: '0.75rem' }}>
             © 2026 SkinAI Project Team. Built with React & FastAPI.
           </p>
        </div>

      </div>
    </footer>
  );
}
