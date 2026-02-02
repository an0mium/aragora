'use client';

import { useCallback, useEffect, useState } from 'react';

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

const STORAGE_KEY = 'aragora_install_prompt';

export function InstallPrompt({
  showDelay = 30000,
  dismissalCooldownDays = 7,
}: {
  showDelay?: number;
  dismissalCooldownDays?: number;
}) {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [showBanner, setShowBanner] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  useEffect(() => {
    const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
    if (isStandalone) {
      setIsInstalled(true);
      return;
    }

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        if (data.installed) {
          setIsInstalled(true);
          return;
        }
        if (data.dismissedAt) {
          const daysSince = (Date.now() - data.dismissedAt) / (24 * 60 * 60 * 1000);
          if (daysSince < dismissalCooldownDays) return;
        }
      }
    } catch {
      // Ignore
    }

    const handler = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
      setTimeout(() => setShowBanner(true), showDelay);
    };

    window.addEventListener('beforeinstallprompt', handler);
    window.addEventListener('appinstalled', () => {
      setIsInstalled(true);
      setShowBanner(false);
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ installed: true }));
    });

    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, [showDelay, dismissalCooldownDays]);

  const handleInstall = useCallback(async () => {
    if (!deferredPrompt) return;
    await deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    if (outcome === 'accepted') {
      setIsInstalled(true);
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ installed: true }));
    }
    setDeferredPrompt(null);
    setShowBanner(false);
  }, [deferredPrompt]);

  const handleDismiss = useCallback(() => {
    setShowBanner(false);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ dismissedAt: Date.now() }));
  }, []);

  if (isInstalled || !showBanner || !deferredPrompt) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 bg-surface border-t border-border p-4 pb-[calc(1rem+env(safe-area-inset-bottom,0px))] animate-slide-in-from-bottom">
      <div className="max-w-lg mx-auto flex items-center gap-4">
        <div className="flex-shrink-0 w-12 h-12 bg-accent/10 rounded-xl flex items-center justify-center text-2xl">
          ⚡
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-text truncate">Install Aragora</h3>
          <p className="text-xs text-text-muted truncate">Quick access from your home screen</p>
        </div>
        <div className="flex-shrink-0 flex items-center gap-2">
          <button
            type="button"
            onClick={handleDismiss}
            className="px-3 py-2 text-sm text-text-muted hover:text-text transition-colors"
          >
            Later
          </button>
          <button
            type="button"
            onClick={handleInstall}
            className="px-4 py-2 text-sm font-medium bg-accent text-bg rounded-lg hover:bg-accent/90 transition-colors"
          >
            Install
          </button>
        </div>
      </div>
    </div>
  );
}

export default InstallPrompt;
