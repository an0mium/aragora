'use client';

import { useState, useEffect } from 'react';
import { API_BASE_URL } from '@/config';

export type BackendType = 'production' | 'development';

interface BackendConfig {
  api: string;
  ws: string;
  label: string;
  description: string;
  fallbackApi?: string;
  fallbackWs?: string;
}

export const BACKENDS: Record<BackendType, BackendConfig> = {
  production: {
    api: 'https://api.aragora.ai',
    ws: 'wss://api.aragora.ai/ws',
    label: 'PROD',
    description: 'AWS Lightsail (always-on)',
  },
  development: {
    api: 'https://api-dev.aragora.ai',
    ws: 'wss://api-dev.aragora.ai/ws',
    label: 'DEV',
    description: 'Local Mac (via tunnel or localhost)',
    fallbackApi: 'http://localhost:8080',
    fallbackWs: 'ws://localhost:8765',
  },
};

const STORAGE_KEY = 'aragora-backend';

interface BackendSelectorProps {
  onChange?: (backend: BackendType, config: BackendConfig) => void;
  compact?: boolean;
}

export function BackendSelector({ onChange, compact = false }: BackendSelectorProps) {
  const [selected, setSelected] = useState<BackendType>('production');
  const [devAvailable, setDevAvailable] = useState<boolean | null>(null);
  const [devSource, setDevSource] = useState<'tunnel' | 'localhost' | null>(null);

  // Load saved preference
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as BackendType;
    if (saved && BACKENDS[saved]) {
      setSelected(saved);
      onChange?.(saved, BACKENDS[saved]);
    }
  }, [onChange]);

  // Check if dev backend is available (try tunnel first, then localhost)
  useEffect(() => {
    const checkEndpoint = async (url: string): Promise<boolean> => {
      try {
        const res = await fetch(`${url}/api/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(3000),
        });
        return res.ok || res.status === 405;
      } catch {
        return false;
      }
    };

    const checkDev = async () => {
      // Try tunnel first
      const tunnelOk = await checkEndpoint(BACKENDS.development.api);
      if (tunnelOk) {
        setDevAvailable(true);
        setDevSource('tunnel');
        return;
      }

      // Try localhost fallback
      if (BACKENDS.development.fallbackApi) {
        const localhostOk = await checkEndpoint(BACKENDS.development.fallbackApi);
        if (localhostOk) {
          setDevAvailable(true);
          setDevSource('localhost');
          return;
        }
      }

      setDevAvailable(false);
      setDevSource(null);
    };

    checkDev();
    const interval = setInterval(checkDev, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  const handleSelect = (backend: BackendType) => {
    setSelected(backend);
    localStorage.setItem(STORAGE_KEY, backend);

    // For development, use localhost URLs if that's what's available
    let config = BACKENDS[backend];
    if (backend === 'development' && devSource === 'localhost' && config.fallbackApi && config.fallbackWs) {
      config = {
        ...config,
        api: config.fallbackApi,
        ws: config.fallbackWs,
      };
    }
    onChange?.(backend, config);
  };

  if (compact) {
    return (
      <div className="flex items-center gap-1 font-mono text-xs">
        <button
          onClick={() => handleSelect('production')}
          className={`px-2 py-1 border transition-colors ${
            selected === 'production'
              ? 'bg-acid-green text-bg border-acid-green'
              : 'text-text-muted border-border hover:text-acid-green hover:border-acid-green/50'
          }`}
          title={BACKENDS.production.description}
        >
          PROD
        </button>
        <button
          onClick={() => handleSelect('development')}
          disabled={devAvailable === false}
          className={`px-2 py-1 border transition-colors ${
            selected === 'development'
              ? 'bg-acid-cyan text-bg border-acid-cyan'
              : devAvailable === false
              ? 'text-text-muted/30 border-border/30 cursor-not-allowed'
              : 'text-text-muted border-border hover:text-acid-cyan hover:border-acid-cyan/50'
          }`}
          title={
            devAvailable === false
              ? 'Dev server offline'
              : devSource === 'localhost'
              ? 'Connected via localhost'
              : BACKENDS.development.description
          }
        >
          DEV
          {devAvailable === false && <span className="ml-1 text-warning">●</span>}
          {devSource === 'localhost' && <span className="ml-1 text-[10px]">L</span>}
        </button>
      </div>
    );
  }

  return (
    <div className="border border-acid-green/30 p-3 bg-surface/50">
      <div className="text-xs text-text-muted mb-2 font-mono">API BACKEND</div>
      <div className="flex gap-2">
        {(Object.entries(BACKENDS) as [BackendType, BackendConfig][]).map(([key, config]) => {
          const isSelected = selected === key;
          const isDisabled = key === 'development' && devAvailable === false;

          return (
            <button
              key={key}
              onClick={() => !isDisabled && handleSelect(key)}
              disabled={isDisabled}
              className={`flex-1 p-2 border font-mono text-left transition-colors ${
                isSelected
                  ? key === 'production'
                    ? 'bg-acid-green/20 border-acid-green text-acid-green'
                    : 'bg-acid-cyan/20 border-acid-cyan text-acid-cyan'
                  : isDisabled
                  ? 'border-border/30 text-text-muted/30 cursor-not-allowed'
                  : 'border-border text-text-muted hover:border-acid-green/50'
              }`}
            >
              <div className="text-sm font-bold flex items-center gap-2">
                {config.label}
                {isSelected && <span>✓</span>}
                {key === 'development' && devAvailable === false && (
                  <span className="text-warning text-xs">OFFLINE</span>
                )}
                {key === 'development' && devAvailable === true && (
                  <span className="text-success text-xs">●</span>
                )}
                {key === 'development' && devSource === 'localhost' && (
                  <span className="text-acid-cyan text-xs">LOCAL</span>
                )}
              </div>
              <div className="text-[10px] opacity-70">
                {key === 'development' && devSource === 'localhost'
                  ? 'Connected via localhost:8080'
                  : config.description}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function useBackend(): { backend: BackendType; config: BackendConfig } {
  const [backend, setBackend] = useState<BackendType>('production');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY) as BackendType;
    if (saved && BACKENDS[saved]) {
      setBackend(saved);
    }

    // Listen for changes
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        setBackend(e.newValue as BackendType);
      }
    };
    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  return { backend, config: BACKENDS[backend] };
}
