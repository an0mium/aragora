'use client';

import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE_URL } from '@/config';

/**
 * Feature information from the backend feature registry
 */
export interface FeatureInfo {
  name: string;
  description: string;
  available: boolean;
  endpoints?: string[];
  install_hint?: string;
  category?: string;
  reason?: string;  // Reason if unavailable (e.g., "Requires configuration: Set API keys")
}

/**
 * Response from /api/features endpoint
 */
export interface FeaturesResponse {
  available: string[];
  unavailable: string[];
  features: Record<string, FeatureInfo>;
}

interface UseFeaturesState {
  features: FeaturesResponse | null;
  loading: boolean;
  error: string | null;
}

/**
 * React hook for feature availability detection
 *
 * Fetches available features from the backend and provides utilities
 * for checking feature availability. Caches results for the session.
 *
 * @example
 * const { isAvailable, features, loading } = useFeatures();
 *
 * // Check if a feature is available
 * if (isAvailable('pulse')) {
 *   // Show trending topics panel
 * }
 *
 * // Get feature info
 * const pulseInfo = features?.features?.pulse;
 */
export function useFeatures(
  apiBase: string = API_BASE_URL
) {
  const [state, setState] = useState<UseFeaturesState>({
    features: null,
    loading: true,
    error: null,
  });

  // Prevent duplicate fetches
  const fetchedRef = useRef(false);

  useEffect(() => {
    if (fetchedRef.current) return;
    fetchedRef.current = true;

    const fetchFeatures = async () => {
      try {
        const response = await fetch(`${apiBase}/api/features`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!response.ok) {
          throw new Error(`Failed to fetch features: ${response.status}`);
        }

        const data: FeaturesResponse = await response.json();
        setState({ features: data, loading: false, error: null });
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Failed to fetch features';
        setState({ features: null, loading: false, error: errorMessage });
      }
    };

    fetchFeatures();
  }, [apiBase]);

  /**
   * Check if a feature is available
   *
   * Returns true if the feature is in the available list,
   * or true by default if features haven't loaded yet (graceful degradation)
   */
  const isAvailable = useCallback(
    (featureId: string): boolean => {
      if (!state.features) {
        // Default to true for graceful degradation while loading
        return true;
      }
      return state.features.available.includes(featureId);
    },
    [state.features]
  );

  /**
   * Get detailed info about a specific feature
   */
  const getFeatureInfo = useCallback(
    (featureId: string): FeatureInfo | undefined => {
      return state.features?.features?.[featureId];
    },
    [state.features]
  );

  /**
   * Get all available feature IDs
   */
  const getAvailableFeatures = useCallback((): string[] => {
    return state.features?.available ?? [];
  }, [state.features]);

  /**
   * Get all unavailable feature IDs
   */
  const getUnavailableFeatures = useCallback((): string[] => {
    return state.features?.unavailable ?? [];
  }, [state.features]);

  /**
   * Force refetch of features
   */
  const refetch = useCallback(async () => {
    fetchedRef.current = false;
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      const response = await fetch(`${apiBase}/api/features`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch features: ${response.status}`);
      }

      const data: FeaturesResponse = await response.json();
      setState({ features: data, loading: false, error: null });
      fetchedRef.current = true;
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : 'Failed to fetch features';
      setState({ features: null, loading: false, error: errorMessage });
      fetchedRef.current = true;
    }
  }, [apiBase]);

  return {
    ...state,
    isAvailable,
    getFeatureInfo,
    getAvailableFeatures,
    getUnavailableFeatures,
    refetch,
  };
}
