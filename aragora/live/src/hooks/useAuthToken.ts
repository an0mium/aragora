'use client';

import { useState, useEffect } from 'react';

const TOKEN_STORAGE_KEY = 'aragora_auth_token';

export function useAuthToken() {
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    // Check URL parameters first
    const urlParams = new URLSearchParams(window.location.search);
    const urlToken = urlParams.get('token');

    if (urlToken) {
      // Store in sessionStorage for persistence
      sessionStorage.setItem(TOKEN_STORAGE_KEY, urlToken);
      setToken(urlToken);
    } else {
      // Check sessionStorage
      const storedToken = sessionStorage.getItem(TOKEN_STORAGE_KEY);
      if (storedToken) {
        setToken(storedToken);
      }
    }
  }, []);

  const getAuthHeaders = () => {
    if (!token) return {};
    return {
      'Authorization': `Bearer ${token}`,
    };
  };

  const getAuthQueryParam = () => {
    if (!token) return '';
    return `token=${encodeURIComponent(token)}`;
  };

  return {
    token,
    getAuthHeaders,
    getAuthQueryParam,
  };
}