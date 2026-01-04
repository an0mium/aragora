'use client';

import { useState, useEffect } from 'react';

interface BootSequenceProps {
  onComplete: () => void;
  skip?: boolean;
}

const BOOT_LINES = [
  { text: 'ARAGORA SYSTEM v2.0.0', delay: 0, style: 'title' },
  { text: '========================', delay: 100, style: 'divider' },
  { text: '', delay: 200, style: 'normal' },
  { text: '[INIT] Loading kernel modules...', delay: 300, style: 'system' },
  { text: '[OK] Multi-agent debate engine', delay: 500, style: 'success' },
  { text: '[OK] ELO rating system', delay: 700, style: 'success' },
  { text: '[OK] Grounded personas module', delay: 900, style: 'success' },
  { text: '[OK] Relationship tracker', delay: 1100, style: 'success' },
  { text: '', delay: 1200, style: 'normal' },
  { text: '[INIT] Connecting agents...', delay: 1300, style: 'system' },
  { text: '  > Claude (Anthropic).......... READY', delay: 1500, style: 'agent' },
  { text: '  > Gemini (Google)............. READY', delay: 1700, style: 'agent' },
  { text: '  > Codex (OpenAI).............. READY', delay: 1900, style: 'agent' },
  { text: '  > Grok (xAI).................. READY', delay: 2100, style: 'agent' },
  { text: '', delay: 2200, style: 'normal' },
  { text: '[INIT] Starting nomic loop...', delay: 2300, style: 'system' },
  { text: '[OK] WebSocket server active', delay: 2500, style: 'success' },
  { text: '[OK] API endpoints mounted', delay: 2700, style: 'success' },
  { text: '', delay: 2800, style: 'normal' },
  { text: '========================', delay: 2900, style: 'divider' },
  { text: 'SYSTEM READY', delay: 3000, style: 'ready' },
  { text: '', delay: 3100, style: 'normal' },
  { text: 'Press any key to continue...', delay: 3200, style: 'prompt' },
];

export function BootSequence({ onComplete, skip = false }: BootSequenceProps) {
  const [visibleLines, setVisibleLines] = useState<number>(0);
  const [showCursor, setShowCursor] = useState(true);
  const [isComplete, setIsComplete] = useState(false);

  // Skip boot sequence if requested
  useEffect(() => {
    if (skip) {
      onComplete();
    }
  }, [skip, onComplete]);

  // Reveal lines progressively
  useEffect(() => {
    if (skip) return;

    const timers: NodeJS.Timeout[] = [];

    BOOT_LINES.forEach((line, index) => {
      const timer = setTimeout(() => {
        setVisibleLines(index + 1);
        if (index === BOOT_LINES.length - 1) {
          setIsComplete(true);
        }
      }, line.delay);
      timers.push(timer);
    });

    return () => timers.forEach(clearTimeout);
  }, [skip]);

  // Cursor blink
  useEffect(() => {
    const interval = setInterval(() => {
      setShowCursor((prev) => !prev);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  // Handle keypress or click to continue
  useEffect(() => {
    if (!isComplete) return;

    const handleInteraction = () => {
      onComplete();
    };

    window.addEventListener('keydown', handleInteraction);
    window.addEventListener('click', handleInteraction);

    // Auto-continue after 2 seconds
    const autoTimer = setTimeout(handleInteraction, 2000);

    return () => {
      window.removeEventListener('keydown', handleInteraction);
      window.removeEventListener('click', handleInteraction);
      clearTimeout(autoTimer);
    };
  }, [isComplete, onComplete]);

  if (skip) return null;

  const getLineStyle = (style: string) => {
    switch (style) {
      case 'title':
        return 'text-acid-green font-bold text-lg glow-text';
      case 'divider':
        return 'text-acid-green/50';
      case 'system':
        return 'text-acid-cyan';
      case 'success':
        return 'text-acid-green';
      case 'agent':
        return 'text-text';
      case 'ready':
        return 'text-acid-green font-bold glow-text animate-pulse';
      case 'prompt':
        return 'text-acid-yellow';
      default:
        return 'text-text';
    }
  };

  return (
    <div className="fixed inset-0 bg-bg z-50 flex items-center justify-center">
      <div className="max-w-2xl w-full p-8 font-mono text-sm">
        {BOOT_LINES.slice(0, visibleLines).map((line, index) => (
          <div
            key={index}
            className={`${getLineStyle(line.style)} boot-line`}
            style={{ animationDelay: `${index * 0.05}s` }}
          >
            {line.text}
            {index === visibleLines - 1 && !isComplete && (
              <span className={showCursor ? 'opacity-100' : 'opacity-0'}>_</span>
            )}
          </div>
        ))}

        {isComplete && (
          <div className="mt-4">
            <span className="text-acid-green">
              {'>'}
              <span className={showCursor ? 'opacity-100' : 'opacity-0'}>_</span>
            </span>
          </div>
        )}
      </div>

      {/* Scanline overlay */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: `repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0.1),
            rgba(0, 0, 0, 0.1) 1px,
            transparent 1px,
            transparent 2px
          )`,
        }}
      />
    </div>
  );
}

// Mini boot animation for component loading
export function MiniLoader({ text = 'Loading' }: { text?: string }) {
  const [dots, setDots] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev + 1) % 4);
    }, 300);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2 font-mono text-sm text-acid-green">
      <span className="animate-pulse">{'>'}</span>
      <span>{text}</span>
      <span className="w-6">{'.'.repeat(dots)}</span>
    </div>
  );
}
