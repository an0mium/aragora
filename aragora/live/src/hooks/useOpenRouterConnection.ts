'use client';

import { useState, useEffect, useCallback } from 'react';
import { getStoredProviderKeys, storeProviderKeys } from '@/lib/provider-keys';
import { startOpenRouterAuth, fetchKeyInfo, type OpenRouterKeyInfo } from '@/lib/openrouter-pkce';

export interface OpenRouterConnection {
  isConnected: boolean;
  keyInfo: OpenRouterKeyInfo | null;
  connect: () => void;
  disconnect: () => void;
  refreshKeyInfo: () => void;
}

export function useOpenRouterConnection(): OpenRouterConnection {
  const [isConnected, setIsConnected] = useState(false);
  const [keyInfo, setKeyInfo] = useState<OpenRouterKeyInfo | null>(null);

  const checkConnection = useCallback(() => {
    const keys = getStoredProviderKeys();
    setIsConnected(!!keys.openrouter);
    return keys.openrouter;
  }, []);

  // Check initial state
  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  // Listen for cross-tab and same-page storage updates
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === 'aragora_provider_keys' || e.key === null) {
        checkConnection();
      }
    };

    // Custom event for same-tab updates (StorageEvent only fires cross-tab)
    const onCustom = () => checkConnection();

    window.addEventListener('storage', onStorage);
    window.addEventListener('openrouter:updated', onCustom);
    return () => {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener('openrouter:updated', onCustom);
    };
  }, [checkConnection]);

  // Fetch key info when connected
  useEffect(() => {
    if (!isConnected) {
      setKeyInfo(null);
      return;
    }
    const key = getStoredProviderKeys().openrouter;
    if (key) {
      fetchKeyInfo(key).then(setKeyInfo);
    }
  }, [isConnected]);

  const connect = useCallback(() => {
    const callbackUrl = `${window.location.origin}/openrouter/callback/`;
    startOpenRouterAuth(callbackUrl);
  }, []);

  const disconnect = useCallback(() => {
    const keys = getStoredProviderKeys();
    delete keys.openrouter;
    storeProviderKeys(keys);
    setIsConnected(false);
    setKeyInfo(null);
    window.dispatchEvent(new Event('openrouter:updated'));
  }, []);

  const refreshKeyInfo = useCallback(() => {
    const key = getStoredProviderKeys().openrouter;
    if (key) {
      fetchKeyInfo(key).then(setKeyInfo);
    }
  }, []);

  return { isConnected, keyInfo, connect, disconnect, refreshKeyInfo };
}
