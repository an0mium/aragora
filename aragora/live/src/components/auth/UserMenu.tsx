'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { useAuth } from '@/context/AuthContext';

export function UserMenu() {
  const { user, organization, isAuthenticated, isLoading, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (isLoading) {
    return (
      <div className="text-xs font-mono text-text-muted animate-pulse">
        [LOADING...]
      </div>
    );
  }

  if (!isAuthenticated || !user) {
    return (
      <div className="flex items-center gap-3">
        <Link
          href="/auth/login"
          className="text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
        >
          [LOGIN]
        </Link>
        <Link
          href="/auth/register"
          className="text-xs font-mono px-3 py-1 bg-acid-green/10 border border-acid-green/50 text-acid-green hover:bg-acid-green/20 transition-colors"
        >
          [REGISTER]
        </Link>
      </div>
    );
  }

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-mono text-acid-cyan hover:text-acid-green transition-colors"
      >
        <span className="w-6 h-6 rounded-full bg-acid-green/20 border border-acid-green/50 flex items-center justify-center text-acid-green">
          {user.name?.[0]?.toUpperCase() || user.email[0].toUpperCase()}
        </span>
        <span className="hidden sm:inline">{user.name || user.email.split('@')[0]}</span>
        <span className="text-acid-green/50">{isOpen ? '[^]' : '[v]'}</span>
      </button>

      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-64 bg-surface border border-acid-green/30 shadow-lg z-50">
          {/* User Info */}
          <div className="p-4 border-b border-acid-green/20">
            <div className="text-sm font-mono text-text">{user.name || 'Anonymous'}</div>
            <div className="text-xs font-mono text-text-muted truncate">{user.email}</div>
            {organization && (
              <div className="mt-2 text-xs font-mono text-acid-cyan">
                ORG: {organization.name}
                <span className="ml-2 px-1 py-0.5 bg-acid-green/10 text-acid-green uppercase">
                  {organization.tier}
                </span>
              </div>
            )}
          </div>

          {/* Menu Items */}
          <div className="py-2">
            <Link
              href="/billing"
              className="block px-4 py-2 text-xs font-mono text-text-muted hover:bg-acid-green/10 hover:text-acid-green transition-colors"
              onClick={() => setIsOpen(false)}
            >
              [BILLING & USAGE]
            </Link>
            <Link
              href="/settings"
              className="block px-4 py-2 text-xs font-mono text-text-muted hover:bg-acid-green/10 hover:text-acid-green transition-colors"
              onClick={() => setIsOpen(false)}
            >
              [SETTINGS]
            </Link>
            <Link
              href="/api-keys"
              className="block px-4 py-2 text-xs font-mono text-text-muted hover:bg-acid-green/10 hover:text-acid-green transition-colors"
              onClick={() => setIsOpen(false)}
            >
              [API KEYS]
            </Link>
            <Link
              href="/ab-testing"
              className="block px-4 py-2 text-xs font-mono text-text-muted hover:bg-acid-green/10 hover:text-acid-green transition-colors"
              onClick={() => setIsOpen(false)}
            >
              [A/B TESTING]
            </Link>
          </div>

          {/* Logout */}
          <div className="border-t border-acid-green/20 py-2">
            <button
              onClick={() => {
                setIsOpen(false);
                logout();
              }}
              className="w-full px-4 py-2 text-xs font-mono text-warning hover:bg-warning/10 transition-colors text-left"
            >
              [LOGOUT]
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
