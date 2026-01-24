'use client';

import { useState, useEffect } from 'react';
import { useOnboardingStore } from '@/store/onboardingStore';

interface OrganizationStepProps {
  onNext: () => void;
  onBack: () => void;
}

export function OrganizationStep({ onNext, onBack }: OrganizationStepProps) {
  const {
    organizationName,
    organizationSlug,
    teamSize,
    setOrganizationName,
    setOrganizationSlug,
    setTeamSize,
  } = useOnboardingStore();

  const [name, setName] = useState(organizationName);
  const [slug, setSlug] = useState(organizationSlug);
  const [size, setSize] = useState<typeof teamSize>(teamSize);

  // Auto-generate slug from name
  useEffect(() => {
    if (!organizationSlug && name) {
      const generatedSlug = name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-|-$/g, '');
      setSlug(generatedSlug);
    }
  }, [name, organizationSlug]);

  const handleNext = () => {
    setOrganizationName(name);
    setOrganizationSlug(slug);
    setTeamSize(size);
    onNext();
  };

  const isValid = name.length >= 3 && size !== null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-mono text-acid-green mb-2">
          Set up your organization
        </h2>
        <p className="font-mono text-text-muted text-sm">
          This helps us configure the right settings for your team
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block font-mono text-sm text-text-muted mb-2">
            Organization name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Acme Inc."
            className="w-full px-4 py-3 bg-surface border border-acid-green/20 font-mono text-sm text-acid-green placeholder:text-text-muted/50 focus:border-acid-green focus:outline-none"
          />
        </div>

        <div>
          <label className="block font-mono text-sm text-text-muted mb-2">
            Organization URL
          </label>
          <div className="flex items-center">
            <span className="px-3 py-3 bg-surface/50 border border-r-0 border-acid-green/20 font-mono text-sm text-text-muted">
              aragora.ai/
            </span>
            <input
              type="text"
              value={slug}
              onChange={(e) => setSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ''))}
              placeholder="acme"
              className="flex-1 px-4 py-3 bg-surface border border-acid-green/20 font-mono text-sm text-acid-green placeholder:text-text-muted/50 focus:border-acid-green focus:outline-none"
            />
          </div>
        </div>

        <div>
          <label className="block font-mono text-sm text-text-muted mb-2">
            Team size
          </label>
          <div className="grid grid-cols-4 gap-2">
            {(['1-5', '6-15', '16-50', '50+'] as const).map((option) => (
              <button
                key={option}
                onClick={() => setSize(option)}
                className={`py-2 font-mono text-sm border transition-colors ${
                  size === option
                    ? 'border-acid-green bg-acid-green/10 text-acid-green'
                    : 'border-acid-green/20 text-text-muted hover:border-acid-green/50'
                }`}
              >
                {option}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-3 bg-acid-cyan/5 border-l-2 border-acid-cyan">
        <p className="text-xs font-mono text-acid-cyan">
          <span className="text-acid-green">TIP:</span> You can invite team members in the next step
        </p>
      </div>

      <div className="flex gap-3 pt-4">
        <button
          onClick={onBack}
          className="px-4 py-2 font-mono text-sm border border-acid-green/30 text-text-muted hover:border-acid-green hover:text-acid-green transition-colors"
        >
          Back
        </button>
        <div className="flex-1" />
        <button
          onClick={handleNext}
          disabled={!isValid}
          className={`px-6 py-2 font-mono text-sm transition-colors ${
            isValid
              ? 'bg-acid-green text-bg hover:bg-acid-green/80'
              : 'bg-surface text-text-muted border border-acid-green/20 cursor-not-allowed'
          }`}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

export default OrganizationStep;
