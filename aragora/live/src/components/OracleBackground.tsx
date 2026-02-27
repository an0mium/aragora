'use client';

import { useRef, useEffect, useCallback } from 'react';

const MODE_PARTICLE_COLORS: Record<string, [number, number, number]> = {
  consult: [200, 100, 200],  // magenta
  divine: [96, 165, 250],    // electric blue
  commune: [74, 222, 128],   // emerald green
};

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  opacity: number;
  baseOpacity: number;
}

interface OracleBackgroundProps {
  mode: 'consult' | 'divine' | 'commune';
}

export function OracleBackground({ mode }: OracleBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animRef = useRef<number>(0);
  const currentColorRef = useRef(MODE_PARTICLE_COLORS.consult);
  const targetColorRef = useRef(MODE_PARTICLE_COLORS.consult);
  const colorTransitionRef = useRef(1); // 0 = transitioning, 1 = done

  // Initialize particles
  const initParticles = useCallback((width: number, height: number) => {
    const count = 50;
    const particles: Particle[] = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: 1 + Math.random() * 2,
        opacity: 0.15 + Math.random() * 0.25,
        baseOpacity: 0.15 + Math.random() * 0.25,
      });
    }
    particlesRef.current = particles;
  }, []);

  // Update target color on mode change
  useEffect(() => {
    targetColorRef.current = MODE_PARTICLE_COLORS[mode] || MODE_PARTICLE_COLORS.consult;
    colorTransitionRef.current = 0;
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Check prefers-reduced-motion
    const motionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (motionQuery.matches) {
      // Static gradient fallback
      const ctx = canvas.getContext('2d');
      if (ctx) {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        const [r, g, b] = MODE_PARTICLE_COLORS[mode] || MODE_PARTICLE_COLORS.consult;
        const grad = ctx.createRadialGradient(
          canvas.width / 2, canvas.height / 3, 0,
          canvas.width / 2, canvas.height / 3, canvas.width * 0.6
        );
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.04)`);
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      if (particlesRef.current.length === 0) {
        initParticles(canvas.width, canvas.height);
      }
    };
    resize();
    window.addEventListener('resize', resize);

    const animate = () => {
      const { width, height } = canvas;
      ctx.clearRect(0, 0, width, height);

      // Interpolate color
      if (colorTransitionRef.current < 1) {
        colorTransitionRef.current = Math.min(1, colorTransitionRef.current + 0.008);
        const t = colorTransitionRef.current;
        const [cr, cg, cb] = currentColorRef.current;
        const [tr, tg, tb] = targetColorRef.current;
        currentColorRef.current = [
          cr + (tr - cr) * t,
          cg + (tg - cg) * t,
          cb + (tb - cb) * t,
        ];
      }

      const [r, g, b] = currentColorRef.current;

      for (const p of particlesRef.current) {
        // Brownian drift
        p.vx += (Math.random() - 0.5) * 0.02;
        p.vy += (Math.random() - 0.5) * 0.02;
        // Damping
        p.vx *= 0.99;
        p.vy *= 0.99;

        p.x += p.vx;
        p.y += p.vy;

        // Wrap edges
        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;

        // Breathing opacity
        p.opacity = p.baseOpacity + Math.sin(Date.now() * 0.001 + p.x * 0.01) * 0.08;

        // Draw particle with radial gradient halo
        const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 6);
        grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${p.opacity})`);
        grad.addColorStop(0.4, `rgba(${r}, ${g}, ${b}, ${p.opacity * 0.3})`);
        grad.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`);

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius * 6, 0, Math.PI * 2);
        ctx.fill();

        // Bright core
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${p.opacity * 1.5})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener('resize', resize);
    };
  }, [mode, initParticles]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
