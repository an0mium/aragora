'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
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
  const mountedRef = useRef(true);
  const keyInfoRequestIdRef = useRef(0);

  const checkConnection = useCallback(() => {
    const keys = getStoredProviderKeys();
    setIsConnected(!!keys.openrouter);
    return keys.openrouter;
  }, []);

  // Check initial state
  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

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
  const loadKeyInfo = useCallback(async (key: string | undefined) => {
    const requestId = ++keyInfoRequestIdRef.current;

    if (!key) {
      if (mountedRef.current && requestId === keyInfoRequestIdRef.current) {
        setKeyInfo(null);
      }
      return null;
    }

    try {
      const info = await fetchKeyInfo(key);
      if (mountedRef.current && requestId === keyInfoRequestIdRef.current) {
        setKeyInfo(info);
      }
      return info;
    } catch (error) {
      console.debug('Failed to refresh OpenRouter key info', error);
      if (mountedRef.current && requestId === keyInfoRequestIdRef.current) {
        setKeyInfo(null);
      }
      return null;
    }
  }, []);

  useEffect(() => {
    if (!isConnected) {
      keyInfoRequestIdRef.current += 1;
      setKeyInfo(null);
      return;
    }
    void loadKeyInfo(getStoredProviderKeys().openrouter);
  }, [isConnected, loadKeyInfo]);

  const connect = useCallback(() => {
    const callbackUrl = `${window.location.origin}/openrouter/callback/`;
    startOpenRouterAuth(callbackUrl);
  }, []);

  const disconnect = useCallback(() => {
    const keys = getStoredProviderKeys();
    delete keys.openrouter;
    storeProviderKeys(keys);
    keyInfoRequestIdRef.current += 1;
    setIsConnected(false);
    setKeyInfo(null);
    window.dispatchEvent(new Event('openrouter:updated'));
  }, []);

  const refreshKeyInfo = useCallback(() => {
    void loadKeyInfo(getStoredProviderKeys().openrouter);
  }, [loadKeyInfo]);

  return { isConnected, keyInfo, connect, disconnect, refreshKeyInfo };
}
