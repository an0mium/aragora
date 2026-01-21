'use client';

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import { useSidebar } from '@/context/SidebarContext';
import { useAuth } from '@/context/AuthContext';
import { useProgressiveMode, ProgressiveMode } from '@/context/ProgressiveModeContext';
import { ModeSelector } from '@/components/ui/FeatureCard';
import { useEdgeSwipe, useSwipeGesture } from '@/hooks/useSwipeGesture';

interface NavItem {
  label: string;
  href: string;
  icon?: string;
  requiresAuth?: boolean;
  adminOnly?: boolean;
  minMode?: ProgressiveMode;
}

const accountItems: NavItem[] = [
  { label: 'Login', href: '/auth/login', icon: '>' },
  { label: 'Register', href: '/auth/register', icon: '+' },
];

const authenticatedAccountItems: NavItem[] = [
  { label: 'Settings', href: '/settings', icon: '*', requiresAuth: true },
  { label: 'Billing', href: '/billing', icon: '$', requiresAuth: true },
  { label: 'Organization', href: '/organization', icon: '@', requiresAuth: true },
];

// Use-case focused navigation - "Start" section
// simple: Basic debate creation
// standard: Full debate controls
// advanced: Power features
// expert: All features including admin/dev tools
const startItems: NavItem[] = [
  { label: 'Hub', href: '/hub', icon: '+' },
  { label: 'New Debate', href: '/arena', icon: '!' },
  { label: 'Stress-Test', href: '/gauntlet', icon: '%', minMode: 'standard' },
  { label: 'Code Review', href: '/reviews', icon: '<', minMode: 'standard' },
  { label: 'Document Audit', href: '/audit', icon: '|', minMode: 'standard' },
];

// Browse section - viewing past content
const browseItems: NavItem[] = [
  { label: 'Debates', href: '/debates', icon: '#' },
  { label: 'Knowledge', href: '/knowledge', icon: '?', minMode: 'standard' },
  { label: 'Leaderboard', href: '/leaderboard', icon: '^', minMode: 'standard' },
  { label: 'Agents', href: '/agents', icon: '&', minMode: 'standard' },
  { label: 'Gallery', href: '/gallery', icon: '*' },
];

// Tools section - management and configuration
const toolsItems: NavItem[] = [
  { label: 'Documents', href: '/documents', icon: ']', minMode: 'standard' },
  { label: 'Workflows', href: '/workflows', icon: '>', minMode: 'advanced' },
  { label: 'Connectors', href: '/connectors', icon: '<', minMode: 'advanced' },
  { label: 'Analytics', href: '/analytics', icon: '~', minMode: 'advanced' },
  { label: 'Templates', href: '/templates', icon: '[', minMode: 'advanced' },
  { label: 'Autonomous', href: '/autonomous', icon: '!', minMode: 'advanced' },
];

// Advanced section - power user features
const advancedItems: NavItem[] = [
  { label: 'Genesis', href: '/genesis', icon: '@', minMode: 'expert' },
  { label: 'Memory', href: '/memory', icon: '=', minMode: 'advanced' },
  { label: 'Introspection', href: '/introspection', icon: '?', minMode: 'expert' },
  { label: 'Verticals', href: '/verticals', icon: '/', minMode: 'advanced' },
  { label: 'Integrations', href: '/integrations', icon: ':', minMode: 'advanced' },
  { label: 'Agent Network', href: '/network', icon: '~', minMode: 'advanced' },
  { label: 'Capability Probe', href: '/probe', icon: '^', minMode: 'expert' },
  { label: 'Red Team', href: '/red-team', icon: '!', minMode: 'advanced' },
  { label: 'Op Modes', href: '/modes', icon: '#', minMode: 'expert' },
];

const adminNavItems: NavItem[] = [
  { label: 'Admin Dashboard', href: '/admin', icon: '!', adminOnly: true },
  { label: 'User Management', href: '/admin/users', icon: '@', adminOnly: true },
  { label: 'Organizations', href: '/admin/organizations', icon: '#', adminOnly: true },
];

export function Sidebar() {
  const { isOpen, close, open } = useSidebar();
  const { isAuthenticated, user, logout } = useAuth();
  const { isFeatureVisible, modeLabel } = useProgressiveMode();
  const sidebarRef = useRef<HTMLDivElement>(null);
  const firstFocusableRef = useRef<HTMLButtonElement>(null);

  const isAdmin = user?.role === 'admin';

  // Edge swipe to open sidebar (from left edge of screen)
  useEdgeSwipe({
    edge: 'left',
    onSwipe: open,
    edgeWidth: 20,
    threshold: 50,
    enabled: !isOpen, // Only enable when sidebar is closed
  });

  // Swipe gesture on sidebar to close (swipe left)
  const swipeRef = useSwipeGesture<HTMLDivElement>({
    onSwipeLeft: close,
    threshold: 50,
    enabled: isOpen,
  });

  // Combine refs
  const combinedRef = (el: HTMLDivElement | null) => {
    (sidebarRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
    (swipeRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
  };

  // Focus trap
  useEffect(() => {
    if (isOpen && firstFocusableRef.current) {
      firstFocusableRef.current.focus();
    }
  }, [isOpen]);

  // Close on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (sidebarRef.current && !sidebarRef.current.contains(e.target as Node)) {
        close();
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, close]);

  const handleLogout = async () => {
    await logout();
    close();
  };

  const renderNavSection = (title: string, items: NavItem[], sectionMinMode?: ProgressiveMode) => {
    // Don't show section at all if user's mode is below section minimum
    if (sectionMinMode && !isFeatureVisible(sectionMinMode)) {
      return null;
    }

    const filteredItems = items.filter(item => {
      if (item.requiresAuth && !isAuthenticated) return false;
      if (item.adminOnly && !isAdmin) return false;
      if (item.minMode && !isFeatureVisible(item.minMode)) return false;
      return true;
    });

    if (filteredItems.length === 0) return null;

    return (
      <div className="mb-6">
        <h3 className="text-acid-cyan text-xs uppercase tracking-wider mb-2 px-2">
          {title}
        </h3>
        <nav>
          {filteredItems.map(item => (
            <Link
              key={item.href}
              href={item.href}
              onClick={close}
              className="flex items-center gap-2 px-2 py-2 text-acid-green hover:bg-acid-green/10 hover:text-acid-cyan transition-colors font-mono text-sm"
            >
              {item.icon && (
                <span className="w-4 text-center text-acid-green/70">{item.icon}</span>
              )}
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    );
  };

  return (
    <>
      {/* Backdrop overlay - solid in light mode, translucent with blur in dark mode */}
      <div
        className={`fixed inset-0 z-40 transition-opacity duration-300 bg-black/70 backdrop-blur-sm ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        data-theme-backdrop
        aria-hidden="true"
      />

      {/* Sidebar panel - fully opaque background */}
      <div
        ref={combinedRef}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className={`fixed top-0 left-0 h-full w-72 sm:w-72 bg-bg border-r border-acid-green/30 z-50 transform transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-acid-green/30">
          <span className="text-acid-green font-mono font-bold text-lg">[MENU]</span>
          <button
            ref={firstFocusableRef}
            onClick={close}
            className="text-acid-green hover:text-acid-cyan transition-colors p-1 focus:outline-none focus:ring-2 focus:ring-acid-green/50 rounded"
            aria-label="Close menu"
          >
            <span className="text-xl">&times;</span>
          </button>
        </div>

        {/* Navigation content */}
        <div className="overflow-y-auto h-[calc(100%-8rem)] p-4">
          {/* Account section */}
          {!isAuthenticated ? (
            renderNavSection('Account', accountItems)
          ) : (
            <>
              {/* User info */}
              <div className="mb-6 p-3 bg-acid-green/5 border border-acid-green/20 rounded">
                <div className="text-acid-green font-mono text-sm truncate">
                  {user?.email || 'User'}
                </div>
                <div className="text-acid-green/60 text-xs mt-1">
                  {user?.role || 'member'}
                </div>
              </div>
              {renderNavSection('Account', authenticatedAccountItems)}
            </>
          )}

          {/* Mode selector */}
          <div className="mb-6 p-2">
            <h3 className="text-acid-cyan text-xs uppercase tracking-wider mb-2 px-2">
              Mode: {modeLabel}
            </h3>
            <ModeSelector compact />
          </div>

          {/* Start - Use cases */}
          {renderNavSection('Start', startItems)}

          {/* Browse - View past content */}
          {renderNavSection('Browse', browseItems)}

          {/* Tools - Management */}
          {renderNavSection('Tools', toolsItems, 'standard')}

          {/* Advanced - Power features */}
          {renderNavSection('Advanced', advancedItems, 'advanced')}

          {/* Admin section */}
          {isAdmin && renderNavSection('Admin', adminNavItems, 'expert')}
        </div>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-acid-green/30 bg-bg">
          {isAuthenticated ? (
            <button
              onClick={handleLogout}
              className="w-full px-4 py-2 text-crimson hover:bg-crimson/10 transition-colors font-mono text-sm border border-crimson/30 rounded"
            >
              Logout
            </button>
          ) : (
            <div className="text-center text-acid-green/50 text-xs font-mono">
              ARAGORA // LIVE
            </div>
          )}
        </div>
      </div>
    </>
  );
}
