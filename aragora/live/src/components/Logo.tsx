'use client';

import Image from 'next/image';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  onClick?: () => void;
  className?: string;
}

const sizes = {
  sm: 24,
  md: 32,
  lg: 48,
};

export function Logo({ size = 'md', onClick, className = '' }: LogoProps) {
  const dimension = sizes[size];

  return (
    <button
      onClick={onClick}
      className={`flex-shrink-0 hover:opacity-80 transition-opacity focus:outline-none focus:ring-2 focus:ring-acid-green/50 rounded ${className}`}
      aria-label="Aragora menu"
      type="button"
    >
      <Image
        src="/aragora-logo.png"
        alt="Aragora"
        width={dimension}
        height={dimension}
        className="drop-shadow-[0_0_8px_rgba(57,255,20,0.3)]"
        priority
      />
    </button>
  );
}
