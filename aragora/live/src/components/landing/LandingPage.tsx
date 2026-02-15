'use client';

import { useState, useCallback } from 'react';
import { Header } from './Header';
import { HeroSection } from './HeroSection';
import { WhyAragoraSection } from './WhyAragoraSection';
import { DebateProtocolSection } from './DebateProtocolSection';
import { CapabilitiesSection } from './CapabilitiesSection';
import { TrustSection } from './TrustSection';
import { Footer } from './Footer';
import type { LandingPageProps } from './types';

export function LandingPage({ apiBase, onDebateStarted }: LandingPageProps) {
  const [error, setError] = useState<string | null>(null);
  const [activeDebateId, setActiveDebateId] = useState<string | null>(null);
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);

  const handleDebateStarted = useCallback(
    (debateId: string, question: string) => {
      setActiveDebateId(debateId);
      setActiveQuestion(question);
      onDebateStarted(debateId);
    },
    [onDebateStarted]
  );

  const handleError = useCallback((err: string) => {
    setError(err);
    setTimeout(() => setError(null), 5000);
  }, []);

  const handleDismissError = useCallback(() => {
    setError(null);
  }, []);

  return (
    <>
      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col">
        <Header />
        <HeroSection
          error={error}
          activeDebateId={activeDebateId}
          activeQuestion={activeQuestion}
          apiBase={apiBase}
          onDismissError={handleDismissError}
          onDebateStarted={handleDebateStarted}
          onError={handleError}
        />
        <WhyAragoraSection />
        <DebateProtocolSection />
        <CapabilitiesSection />
        <TrustSection />
        <Footer />
      </main>
    </>
  );
}
