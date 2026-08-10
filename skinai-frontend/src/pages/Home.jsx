import React, { useState, useCallback } from 'react';
import Hero from '../components/Hero';
import Pipeline from '../components/Pipeline';
import ResultsCard from '../components/ResultsCard';
import DiseaseGrid from '../components/DiseaseGrid';
import TeamSection from '../components/TeamSection';
import { predictImage } from '../api';

export default function Home({ apiOnline }) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [pipelineResult, setPipelineResult] = useState(null);
  const [activeStep, setActiveStep] = useState(-1);
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [patientInfo, setPatientInfo] = useState({ name: '', gender: '', age: '' });

  const animatePipeline = useCallback((res) => {
    setPipelineResult(res.pipeline_images || null);
    let step = 0;
    const tick = () => {
      setActiveStep(step);
      step++;
      if (step < 6) setTimeout(tick, 250);
      else setTimeout(() => setActiveStep(6), 250);
    };
    tick();
  }, []);

  const handleAnalyze = useCallback(async (file) => {
    if (!file) return;
    setLoading(true);
    setIsProcessing(true);
    setError('');
    setResult(null);
    setActiveStep(-1);
    setPipelineResult(null);
    setImageUrl(URL.createObjectURL(file));

    try {
      const res = await predictImage(file);
      animatePipeline(res);
      // Wait for pipeline to finish before showing results
      setTimeout(() => {
        setResult(res);
        setIsProcessing(false); 
        setLoading(false);
      }, 7 * 250 + 500);
    } catch (err) {
      setError(err.message || 'Prediction failed');
      setIsProcessing(false);
      setLoading(false);
    }
  }, [animatePipeline]);

  const handleReset = () => {
    setResult(null);
    setImageUrl(null);
    setPipelineResult(null);
    setActiveStep(-1);
    setError('');
    setIsProcessing(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <main>
      <Hero 
        onAnalyze={handleAnalyze} 
        loading={loading} 
        patientInfo={patientInfo} 
        setPatientInfo={setPatientInfo} 
      />
      
      {error && (
        <div style={{ maxWidth: 900, margin: '1rem auto', padding: '0 2rem' }}>
          <div style={{ background: '#FEF2F2', border: '1px solid #FCA5A5', color: '#B91C1C', padding: '1rem', borderRadius: '12px', fontSize: '0.9rem' }}>
            {error}
          </div>
        </div>
      )}

      <Pipeline 
        pipelineResult={pipelineResult} 
        activeStep={activeStep} 
        isVisible={isProcessing || result !== null} 
      />

      {result && !isProcessing && (
        <ResultsCard 
          result={result} 
          imageUrl={imageUrl} 
          onReset={handleReset} 
          patientInfo={patientInfo} 
        />
      )}

      <DiseaseGrid />
      <TeamSection />
    </main>
  );
}
