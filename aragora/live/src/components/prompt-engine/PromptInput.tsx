'use client';

import { useState } from 'react';

interface PromptInputProps {
  onSubmit: (prompt: string, profile: string, sources?: string[]) => Promise<void>;
}

const PROFILES = [
  { value: 'founder', label: 'Founder', desc: 'Quick, auto-execute at 80%' },
  { value: 'cto', label: 'CTO', desc: 'Thorough, auto-execute at 90%' },
  { value: 'business', label: 'Business', desc: 'Thorough, auto-execute at 95%' },
  { value: 'team', label: 'Team', desc: 'Exhaustive, always approve' },
];

export function PromptInput({ onSubmit }: PromptInputProps) {
  const [prompt, setPrompt] = useState('');
  const [profile, setProfile] = useState('founder');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (!prompt.trim() || isSubmitting) return;
    setIsSubmitting(true);
    try {
      await onSubmit(prompt.trim(), profile);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold font-mono text-[var(--acid-green)]">
        Prompt-to-Spec Engine
      </h1>
      <p className="text-sm text-[var(--text-muted)] font-mono">
        Describe what you want to build. The engine will decompose, research, and generate a validated spec.
      </p>
      <textarea
        value={prompt}
        onChange={e => setPrompt(e.target.value)}
        placeholder="make onboarding better..."
        className="w-full h-40 bg-[var(--surface)] border border-[var(--border)] p-4 font-mono text-sm text-[var(--text)] placeholder:text-[var(--text-muted)]/50 focus:border-[var(--acid-green)] focus:outline-none resize-none"
      />
      <div className="flex gap-2">
        {PROFILES.map(p => (
          <button
            key={p.value}
            onClick={() => setProfile(p.value)}
            className={`px-3 py-1.5 font-mono text-xs border ${
              profile === p.value
                ? 'border-[var(--acid-green)] text-[var(--acid-green)] bg-[var(--acid-green)]/10'
                : 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--text-muted)]'
            }`}
            title={p.desc}
          >
            {p.label}
          </button>
        ))}
      </div>
      <button
        onClick={handleSubmit}
        disabled={!prompt.trim() || isSubmitting}
        className="px-6 py-2 bg-[var(--acid-green)] text-[var(--bg)] font-mono font-bold text-sm hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'Starting...' : 'Generate Spec'}
      </button>
    </div>
  );
}
