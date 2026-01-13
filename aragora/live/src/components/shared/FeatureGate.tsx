'use client';

import { type ReactNode } from 'react';
import { isFeatureEnabled, type FeatureName } from '@/lib/featureFlags';

interface FeatureGateProps {
  /** Feature flag name to check */
  feature: FeatureName;
  /** Content to render when feature is enabled */
  children: ReactNode;
  /** Optional fallback when feature is disabled */
  fallback?: ReactNode;
}

/**
 * Conditionally render content based on feature flag status
 *
 * @example
 * <FeatureGate feature="GRAPH_DEBATES">
 *   <GraphDebatePanel />
 * </FeatureGate>
 *
 * @example
 * <FeatureGate
 *   feature="MATRIX_DEBATES"
 *   fallback={<ComingSoonBanner feature="Matrix Debates" />}
 * >
 *   <MatrixDebatePanel />
 * </FeatureGate>
 */
export function FeatureGate({ feature, children, fallback = null }: FeatureGateProps) {
  if (isFeatureEnabled(feature)) {
    return <>{children}</>;
  }
  return <>{fallback}</>;
}

/**
 * Hook to check if a feature is enabled
 * Re-renders when localStorage changes (for runtime toggles)
 */
export function useFeatureFlag(feature: FeatureName): boolean {
  // For SSR safety, always check on client
  if (typeof window === 'undefined') {
    return false;
  }
  return isFeatureEnabled(feature);
}
