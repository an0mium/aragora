'use client';

import { useCallback, useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { LandingPage } from '@/components/LandingPage';
import { useBackend, BACKENDS } from '@/components/BackendSelector';

export default function LandingRoute() {
  const router = useRouter();
  const { config: backendConfig } = useBackend();

  const [apiBase, setApiBase] = useState(BACKENDS.production.api);
  const [wsUrl, setWsUrl] = useState(BACKENDS.production.ws);

  useEffect(() => {
    setApiBase(backendConfig.api);
    setWsUrl(backendConfig.ws);
  }, [backendConfig]);

  const handleDebateStarted = useCallback((debateId: string) => {
    router.push(`/debate/${debateId}`);
  }, [router]);

  const handleEnterDashboard = useCallback(() => {
    router.push('/');
  }, [router]);

  return (
    <LandingPage
      apiBase={apiBase}
      wsUrl={wsUrl}
      onDebateStarted={handleDebateStarted}
      onEnterDashboard={handleEnterDashboard}
    />
  );
}
