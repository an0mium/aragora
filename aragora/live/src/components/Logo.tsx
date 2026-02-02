'use client';

interface LogoProps {
  size?: 'sm' | 'md' | 'lg';
  pixelSize?: number;
  onClick?: () => void;
  className?: string;
}

const sizes = {
  sm: 16,
  md: 24,
  lg: 32,
};

export function Logo({ size = 'md', pixelSize, onClick, className = '' }: LogoProps) {
  const dimension = pixelSize ?? sizes[size];
  const src = dimension <= 16 ? '/favicon-16.png' : dimension <= 32 ? '/favicon-32.png' : '/favicon.ico';

  return (
    <button
      onClick={onClick}
      className={`group flex-shrink-0 transition-all focus:outline-none focus:ring-2 focus:ring-acid-green/50 rounded ${className}`}
      aria-label="Aragora menu"
      type="button"
    >
      <img
        src={src}
        alt="Aragora"
        width={dimension}
        height={dimension}
        className="block transition-all group-hover:drop-shadow-[0_0_10px_rgba(57,255,20,0.7)] group-hover:brightness-110"
      />
    </button>
  );
}
