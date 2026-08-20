import React, { useRef, useEffect } from 'react';

/**
 * Enhanced Clinical & Neural Particle Canvas
 * Supports interactive neural constellation nodes, clinical plus marks (+),
 * diamond prisms (✦), and pulsing synapse sparks.
 */
const FloatingParticlesBackground = ({ 
  count = 18, 
  colors = ['#2563EB', '#10B981', '#38BDF8', '#818CF8'], 
  opacity = 0.35, 
  speed = 0.3,
  connectLines = true,
  maxLineDistance = 90,
  variant = 'neural' // 'neural' | 'bubbles'
}) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let animationFrameId;
    
    const particles = [];
    const shapes = ['node', 'cross', 'diamond', 'spark'];
    
    const resize = () => {
      const parent = canvas.parentElement;
      if (parent) {
        const dpr = window.devicePixelRatio || 1;
        const rect = parent.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = `${rect.width}px`;
        canvas.style.height = `${rect.height}px`;
        ctx.scale(dpr, dpr);
      }
    };

    window.addEventListener('resize', resize);
    resize();

    const parent = canvas.parentElement;
    const w = parent ? parent.clientWidth : 300;
    const h = parent ? parent.clientHeight : 100;

    // Initialize particles
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        size: Math.random() * 4 + 2.5,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
        rot: Math.random() * Math.PI * 2,
        rotSpeed: (Math.random() - 0.5) * 0.02,
        shape: shapes[Math.floor(Math.random() * shapes.length)],
        alpha: Math.random() * 0.5 + 0.5,
        pulse: Math.random() * Math.PI * 2,
        color: colors[Math.floor(Math.random() * colors.length)]
      });
    }

    const draw = () => {
      const width = parent ? parent.clientWidth : 300;
      const height = parent ? parent.clientHeight : 100;
      ctx.clearRect(0, 0, width, height);

      // 1. Draw neural constellation connection lines
      if (connectLines && variant === 'neural') {
        for (let i = 0; i < particles.length; i++) {
          for (let j = i + 1; j < particles.length; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = Math.sqrt(dx * dx + dy * dy);

            if (dist < maxLineDistance) {
              const lineAlpha = (1 - dist / maxLineDistance) * opacity * 0.45;
              ctx.beginPath();
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.strokeStyle = particles[i].color;
              ctx.globalAlpha = lineAlpha;
              ctx.lineWidth = 0.75;
              ctx.stroke();
            }
          }
        }
      }

      // 2. Draw individual geometric / neural particles
      particles.forEach(p => {
        p.pulse += 0.03;
        p.rot += p.rotSpeed;
        const currentAlpha = opacity * p.alpha * (0.8 + 0.2 * Math.sin(p.pulse));

        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = currentAlpha;

        if (variant === 'bubbles') {
          const r = p.size * 4.5;
          // Volumetric 3D bubble with radial depth
          const grad = ctx.createRadialGradient(-r * 0.25, -r * 0.25, r * 0.1, 0, 0, r);
          grad.addColorStop(0, 'rgba(255, 255, 255, 0.7)');
          grad.addColorStop(0.35, p.color + '40');
          grad.addColorStop(0.85, p.color + '90');
          grad.addColorStop(1, p.color + '20');

          ctx.beginPath();
          ctx.arc(0, 0, r, 0, Math.PI * 2);
          ctx.fillStyle = grad;
          ctx.fill();

          // Outer glass ring highlight
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.65)';
          ctx.lineWidth = 0.8;
          ctx.stroke();

          // Specular glint dot (top-left reflection)
          ctx.beginPath();
          ctx.arc(-r * 0.38, -r * 0.38, r * 0.22, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
          ctx.fill();

          // Bottom-right subtle counter-reflection
          ctx.beginPath();
          ctx.arc(r * 0.3, r * 0.3, r * 0.14, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.fill();
        } else if (p.shape === 'cross') {
          // Medical cross / plus symbol (+)
          const arm = p.size * 1.6;
          const thick = 1.2;
          ctx.lineWidth = thick;
          ctx.beginPath();
          ctx.moveTo(-arm, 0);
          ctx.lineTo(arm, 0);
          ctx.moveTo(0, -arm);
          ctx.lineTo(0, arm);
          ctx.stroke();
        } else if (p.shape === 'diamond') {
          // 4-point clinical diamond star (✦)
          const r = p.size * 1.5;
          ctx.beginPath();
          ctx.moveTo(0, -r);
          ctx.lineTo(r * 0.4, -r * 0.3);
          ctx.lineTo(r, 0);
          ctx.lineTo(r * 0.4, r * 0.3);
          ctx.lineTo(0, r);
          ctx.lineTo(-r * 0.4, r * 0.3);
          ctx.lineTo(-r, 0);
          ctx.lineTo(-r * 0.4, -r * 0.3);
          ctx.closePath();
          ctx.fill();
        } else if (p.shape === 'spark') {
          // Glowing neural synapse (dot with pulse ring)
          ctx.beginPath();
          ctx.arc(0, 0, p.size * 0.8, 0, Math.PI * 2);
          ctx.fill();
          
          ctx.beginPath();
          ctx.arc(0, 0, p.size * 2 + Math.sin(p.pulse) * 1.5, 0, Math.PI * 2);
          ctx.lineWidth = 0.8;
          ctx.stroke();
        } else {
          // Sharp neural node
          ctx.beginPath();
          ctx.arc(0, 0, p.size, 0, Math.PI * 2);
          ctx.fill();
        }

        ctx.restore();

        // Update movement
        p.x += p.vx;
        p.y += p.vy;

        // Wrap around boundaries
        const pad = 15;
        if (p.x < -pad) p.x = width + pad;
        if (p.x > width + pad) p.x = -pad;
        if (p.y < -pad) p.y = height + pad;
        if (p.y > height + pad) p.y = -pad;
      });

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animationFrameId);
    };
  }, [count, colors, opacity, speed, connectLines, maxLineDistance, variant]);

  return (
    <canvas 
      ref={canvasRef} 
      style={{ 
        position: 'absolute', 
        inset: 0, 
        width: '100%', 
        height: '100%', 
        zIndex: 0, 
        pointerEvents: 'none' 
      }} 
    />
  );
};

export default FloatingParticlesBackground;
