'use client';

import { ReactNode } from 'react';

interface AsciiCardProps {
  children: ReactNode;
  title?: string;
  variant?: 'default' | 'terminal' | 'success' | 'warning' | 'danger';
  className?: string;
  glow?: boolean;
}

// Box-drawing characters for different styles
const CORNERS = {
  default: { tl: '+', tr: '+', bl: '+', br: '+', h: '-', v: '|' },
  rounded: { tl: '.', tr: '.', bl: "'", br: "'", h: '-', v: '|' },
  double: { tl: '#', tr: '#', bl: '#', br: '#', h: '=', v: '|' },
  unicode: { tl: '\u250C', tr: '\u2510', bl: '\u2514', br: '\u2518', h: '\u2500', v: '\u2502' },
};

export function AsciiCard({
  children,
  title,
  variant = 'default',
  className = '',
  glow = false,
}: AsciiCardProps) {
  const variantStyles: Record<string, string> = {
    default: 'border-acid-green/50',
    terminal: 'border-acid-green',
    success: 'border-acid-green',
    warning: 'border-acid-yellow',
    danger: 'border-crimson',
  };

  const glowStyles: Record<string, string> = {
    default: 'shadow-[0_0_10px_rgba(57,255,20,0.1)]',
    terminal: 'shadow-[0_0_15px_rgba(57,255,20,0.2)]',
    success: 'shadow-[0_0_15px_rgba(57,255,20,0.3)]',
    warning: 'shadow-[0_0_15px_rgba(255,255,0,0.2)]',
    danger: 'shadow-[0_0_15px_rgba(255,0,64,0.2)]',
  };

  return (
    <div
      className={`
        relative bg-surface border ${variantStyles[variant]}
        ${glow ? glowStyles[variant] : ''}
        ${className}
      `}
    >
      {/* Title bar */}
      {title && (
        <div className="border-b border-inherit px-3 py-1.5 flex items-center gap-2">
          <span className="text-acid-green font-mono text-xs">
            {'>'} {title.toUpperCase()}
          </span>
          <span className="flex-1 border-b border-dashed border-acid-green/30" />
        </div>
      )}

      {/* Content */}
      <div className="p-3">{children}</div>
    </div>
  );
}

// Terminal-style header for sections
export function AsciiHeader({
  children,
  level = 2,
}: {
  children: ReactNode;
  level?: 1 | 2 | 3;
}) {
  const sizes = {
    1: 'text-lg',
    2: 'text-base',
    3: 'text-sm',
  };

  return (
    <div className="flex items-center gap-2 mb-3">
      <span className="text-acid-green font-mono">[</span>
      <span className={`text-text font-bold ${sizes[level]}`}>{children}</span>
      <span className="text-acid-green font-mono">]</span>
      <span className="flex-1 border-b border-acid-green/30" />
    </div>
  );
}

// Progress bar with ASCII aesthetic
export function AsciiProgress({
  value,
  max = 100,
  label,
  variant = 'default',
  showPercentage = true,
}: {
  value: number;
  max?: number;
  label?: string;
  variant?: 'default' | 'success' | 'warning' | 'danger';
  showPercentage?: boolean;
}) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  const filled = Math.floor(percentage / 5); // 20 chars total
  const empty = 20 - filled;

  const variantColors: Record<string, string> = {
    default: 'text-acid-green',
    success: 'text-acid-green',
    warning: 'text-acid-yellow',
    danger: 'text-crimson',
  };

  return (
    <div className="font-mono text-xs">
      {label && (
        <div className="flex justify-between mb-1">
          <span className="text-text-muted">{label}</span>
          {showPercentage && (
            <span className={variantColors[variant]}>{percentage.toFixed(0)}%</span>
          )}
        </div>
      )}
      <div className="flex items-center gap-1">
        <span className="text-acid-green/50">[</span>
        <span className={variantColors[variant]}>
          {'#'.repeat(filled)}
          {'-'.repeat(empty)}
        </span>
        <span className="text-acid-green/50">]</span>
      </div>
    </div>
  );
}

// Loading spinner with ASCII animation
export function AsciiSpinner({ text = 'Loading' }: { text?: string }) {
  const frames = ['|', '/', '-', '\\'];
  const [frame, setFrame] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame((prev) => (prev + 1) % frames.length);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <span className="font-mono text-acid-green">
      {text} <span className="inline-block w-3">{frames[frame]}</span>
    </span>
  );
}

// Hook for useState
import { useState, useEffect } from 'react';

// Typing effect for text
export function TypewriterText({
  text,
  speed = 50,
  className = '',
}: {
  text: string;
  speed?: number;
  className?: string;
}) {
  const [displayed, setDisplayed] = useState('');
  const [showCursor, setShowCursor] = useState(true);

  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.substring(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
      }
    }, speed);
    return () => clearInterval(interval);
  }, [text, speed]);

  useEffect(() => {
    const interval = setInterval(() => {
      setShowCursor((prev) => !prev);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <span className={className}>
      {displayed}
      <span className={`${showCursor ? 'opacity-100' : 'opacity-0'} text-acid-green`}>
        _
      </span>
    </span>
  );
}

// Badge component
export function AsciiBadge({
  children,
  variant = 'default',
}: {
  children: ReactNode;
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
}) {
  const variants: Record<string, string> = {
    default: 'border-text-muted text-text-muted',
    success: 'border-acid-green text-acid-green',
    warning: 'border-acid-yellow text-acid-yellow',
    danger: 'border-crimson text-crimson',
    info: 'border-acid-cyan text-acid-cyan',
  };

  return (
    <span
      className={`
        inline-flex items-center px-1.5 py-0.5
        text-xs font-mono border
        ${variants[variant]}
      `}
    >
      {children}
    </span>
  );
}
