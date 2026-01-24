'use client';

import { useRouter } from 'next/navigation';
import { useCallback } from 'react';
import { Scanlines, CRTVignette } from '@/components/MatrixRain';
import { OnboardingWizardFlow } from '@/components/onboarding';

export default function OnboardingPage() {
  const router = useRouter();

  const handleComplete = useCallback(() => {
    router.push('/');
  }, [router]);

  const handleSkip = useCallback(() => {
    router.push('/');
  }, [router]);

  return (
    <>
      <Scanlines opacity={0.02} />
      <CRTVignette />

      <main className="min-h-screen bg-bg text-text relative z-10">
        <OnboardingWizardFlow
          onComplete={handleComplete}
          onSkip={handleSkip}
        />
      </main>
    </>
  );
}
