'use client';

import { useState, useEffect } from 'react';

export type BackendType = 'production' | 'development';

export interface BackendConfig {
  api: string;
  ws: string;
  controlPlaneWs?: string;
  label: string;
  description: string;
  fallbackApi?: string;
  fallbackWs?: string;
  fallbackControlPlaneWs?: string;
}

export const BACKENDS: Record<BackendType, BackendConfig> = {
  production: {
    api: 'https://api.aragora.ai',
    ws: 'wss://api.aragora.ai/ws',
    controlPlaneWs: 'wss://api.aragora.ai/api/control-plane/stream',
    label: 'PROD',
    description: 'AWS Lightsail (always-on)',
  },
  development: {
    api: 'https://api-dev.aragora.ai',
    ws: 'wss://api-dev.aragora.ai/ws',
    controlPlaneWs: 'wss://api-dev.aragora.ai/api/control-plane/stream',
    label: 'DEV',
    description: 'Local Mac (via tunnel or localhost)',
    fallbackApi: '',
    fallbackWs: 'ws://localhost:8765/ws',
    fallbackControlPlaneWs: 'ws://localhost:8766/api/control-plane/stream',
  },
};

const STORAGE_KEY = 'aragora-backend';
export const BACKEND_CHANGE_EVENT = 'aragora-backend-change';

export function buildHealthCheckUrl(apiBase: string): string {
  const normalizedBase = apiBase.trim().replace(/\/$/, '');
  if (!normalizedBase) {
    // In local same-origin dev, Next rewrites `/api/*` and `trailingSlash: true`
    // turns `/api/health` into a redirect chain. Use the stable path directly.
    return '/api/health/';
  }
  return `${normalizedBase}/api/health`;
}

function isLocalHost(hostname: string | undefined): boolean {
  if (!hostname) return false;
  return (
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname.startsWith('192.168.') ||
    hostname.startsWith('10.') ||
    hostname.endsWith('.local')
  );
}

function getSavedBackend(): BackendType | null {
  if (typeof window === 'undefined') return null;
  const saved = localStorage.getItem(STORAGE_KEY) as BackendType | null;
  return saved && BACKENDS[saved] ? saved : null;
}

function getDefaultBackend(): BackendType {
  const saved = getSavedBackend();
  if (saved) return saved;
  if (typeof window !== 'undefined' && isLocalHost(window.location.hostname)) {
    return 'development';
  }
  return 'production';
}

function resolveBackendConfig(
  backend: BackendType,
  devSource: 'tunnel' | 'localhost' | null,
): BackendConfig {
  const config = BACKENDS[backend];
  if (
    backend === 'development' &&
    devSource === 'localhost' &&
    config.fallbackApi !== undefined &&
    config.fallbackWs
  ) {
    return {
      ...config,
      api: config.fallbackApi,
      ws: config.fallbackWs,
      ...(config.fallbackControlPlaneWs
        ? { controlPlaneWs: config.fallbackControlPlaneWs }
        : {}),
    };
  }
  return config;
}

export function getRuntimeBackendConfig(): { backend: BackendType; config: BackendConfig } {
  const localHost =
    typeof window !== 'undefined' && isLocalHost(window.location.hostname);
  const backend = getDefaultBackend();
  return {
    backend,
    config: resolveBackendConfig(backend, localHost ? 'localhost' : null),
  };
}

interface BackendSelectorProps {
  onChange?: (backend: BackendType, config: BackendConfig) => void;
  compact?: boolean;
}

export function BackendSelector({ onChange, compact = false }: BackendSelectorProps) {
  const localHost = typeof window !== 'undefined' && isLocalHost(window.location.hostname);
  const [selected, setSelected] = useState<BackendType>(getDefaultBackend);
  const [devAvailable, setDevAvailable] = useState<boolean | null>(localHost ? true : null);
  const [devSource, setDevSource] = useState<'tunnel' | 'localhost' | null>(
    localHost ? 'localhost' : null,
  );

  // Load saved preference
  useEffect(() => {
    const saved = getSavedBackend();
    if (saved) {
      setSelected(saved);
      onChange?.(saved, resolveBackendConfig(saved, devSource));
    } else if (localHost) {
      onChange?.('development', resolveBackendConfig('development', 'localhost'));
    }
  }, [devSource, localHost, onChange]);

  // Check if dev backend is available (try tunnel first, then localhost)
  useEffect(() => {
    const checkEndpoint = async (url: string): Promise<boolean> => {
      try {
        const res = await fetch(buildHealthCheckUrl(url), {
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

    if (localHost) {
      setDevAvailable(true);
      setDevSource('localhost');
    }

    checkDev();
    const interval = setInterval(checkDev, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, [localHost]);

  const handleSelect = (backend: BackendType) => {
    setSelected(backend);
    localStorage.setItem(STORAGE_KEY, backend);
    window.dispatchEvent(new CustomEvent<BackendType>(BACKEND_CHANGE_EVENT, { detail: backend }));
    onChange?.(backend, resolveBackendConfig(backend, devSource));
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
  const localHost =
    typeof window !== 'undefined' && isLocalHost(window.location.hostname);
  const [backend, setBackend] = useState<BackendType>(() => getRuntimeBackendConfig().backend);

  useEffect(() => {
    const saved = getSavedBackend();
    if (saved) {
      setBackend(saved);
    } else if (localHost) {
      setBackend('development');
    }

    // Listen for changes
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        setBackend(e.newValue as BackendType);
      }
    };
    const handleBackendChange = (event: Event) => {
      const nextBackend = (event as CustomEvent<BackendType>).detail;
      if (nextBackend && BACKENDS[nextBackend]) {
        setBackend(nextBackend);
      }
    };
    window.addEventListener('storage', handleStorage);
    window.addEventListener(BACKEND_CHANGE_EVENT, handleBackendChange as EventListener);
    return () => {
      window.removeEventListener('storage', handleStorage);
      window.removeEventListener(BACKEND_CHANGE_EVENT, handleBackendChange as EventListener);
    };
  }, [localHost]);

  return {
    backend,
    config: resolveBackendConfig(backend, localHost ? 'localhost' : null),
  };
}
