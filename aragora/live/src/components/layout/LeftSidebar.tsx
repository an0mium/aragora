'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useLayout } from '@/context/LayoutContext';
import { useAuth } from '@/context/AuthContext';
import { useProgressiveMode, ProgressiveMode } from '@/context/ProgressiveModeContext';
import { ModeSelector } from '@/components/ui/FeatureCard';

interface NavItem {
  label: string;
  href: string;
  icon: string;
  minMode?: ProgressiveMode;
  requiresAuth?: boolean;
  adminOnly?: boolean;
}

// Quick action items - always visible at top
const quickActions: NavItem[] = [
  { label: 'New Debate', href: '/arena', icon: '+' },
  { label: 'Stress Test', href: '/gauntlet', icon: '%' },
];

// Primary navigation items
const primaryNav: NavItem[] = [
  { label: 'Home', href: '/', icon: '≡' },
  { label: 'Debates', href: '/debates', icon: '⌘' },
  { label: 'Knowledge', href: '/knowledge', icon: '?' },
  { label: 'Agents', href: '/agents', icon: '&' },
  { label: 'Analytics', href: '/analytics', icon: '~', minMode: 'standard' },
  { label: 'Settings', href: '/settings', icon: '*', requiresAuth: true },
];

// Browse section items - collapsed by default
const browseItems: NavItem[] = [
  { label: 'Gallery', href: '/gallery', icon: '✦' },
  { label: 'Leaderboard', href: '/leaderboard', icon: '^' },
  { label: 'Tournaments', href: '/tournaments', icon: '⊕', minMode: 'standard' },
  { label: 'Reviews', href: '/reviews', icon: '<' },
];

// Tools section - progressive disclosure
const toolsItems: NavItem[] = [
  { label: 'Documents', href: '/documents', icon: ']' },
  { label: 'Workflows', href: '/workflows', icon: '>', minMode: 'advanced' },
  { label: 'Connectors', href: '/connectors', icon: '<', minMode: 'advanced' },
  { label: 'Templates', href: '/templates', icon: '[', minMode: 'advanced' },
];

// Advanced section - expert users
const advancedItems: NavItem[] = [
  { label: 'Genesis', href: '/genesis', icon: '@', minMode: 'expert' },
  { label: 'Memory', href: '/memory', icon: '=', minMode: 'advanced' },
  { label: 'Introspection', href: '/introspection', icon: '⊙', minMode: 'expert' },
];

export function LeftSidebar() {
  const pathname = usePathname();
  const {
    leftSidebarOpen,
    leftSidebarCollapsed,
    closeLeftSidebar,
    setLeftSidebarCollapsed,
    isMobile,
    leftSidebarWidth,
  } = useLayout();
  const { isAuthenticated, user } = useAuth();
  const { isFeatureVisible } = useProgressiveMode();

  // Don't render on desktop if closed, but always render for mobile (as overlay)
  if (!isMobile && !leftSidebarOpen) {
    return null;
  }

  const isAdmin = user?.role === 'admin';

  const filterItems = (items: NavItem[]) =>
    items.filter(item => {
      if (item.requiresAuth && !isAuthenticated) return false;
      if (item.adminOnly && !isAdmin) return false;
      if (item.minMode && !isFeatureVisible(item.minMode)) return false;
      return true;
    });

  const renderNavItem = (item: NavItem) => {
    const isActive = pathname === item.href || pathname?.startsWith(item.href + '/');

    return (
      <Link
        key={item.href}
        href={item.href}
        onClick={() => isMobile && closeLeftSidebar()}
        className={`
          flex items-center gap-3 px-3 py-2 rounded-md transition-colors
          ${isActive
            ? 'bg-[var(--acid-green)]/10 text-[var(--acid-green)]'
            : 'text-[var(--text-muted)] hover:bg-[var(--surface-elevated)] hover:text-[var(--text)]'
          }
        `}
        title={leftSidebarCollapsed ? item.label : undefined}
      >
        <span className="font-mono text-lg w-6 text-center">{item.icon}</span>
        {!leftSidebarCollapsed && (
          <span className="text-sm font-medium">{item.label}</span>
        )}
      </Link>
    );
  };

  const renderSection = (title: string, items: NavItem[], minMode?: ProgressiveMode) => {
    if (minMode && !isFeatureVisible(minMode)) return null;

    const filtered = filterItems(items);
    if (filtered.length === 0) return null;

    return (
      <div className="mb-4">
        {!leftSidebarCollapsed && (
          <div className="px-3 mb-2 text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider">
            {title}
          </div>
        )}
        <nav className="space-y-1">
          {filtered.map(renderNavItem)}
        </nav>
      </div>
    );
  };

  const sidebarContent = (
    <div className="flex flex-col h-full">
      {/* Quick Actions */}
      <div className="p-3 border-b border-[var(--border)]">
        {filterItems(quickActions).map(item => (
          <Link
            key={item.href}
            href={item.href}
            onClick={() => isMobile && closeLeftSidebar()}
            className="flex items-center gap-2 px-3 py-2 mb-1 rounded-md bg-[var(--acid-green)]/10 text-[var(--acid-green)] hover:bg-[var(--acid-green)]/20 transition-colors"
          >
            <span className="font-mono text-lg">{item.icon}</span>
            {!leftSidebarCollapsed && (
              <span className="text-sm font-medium">{item.label}</span>
            )}
          </Link>
        ))}
      </div>

      {/* Scrollable Navigation */}
      <div className="flex-1 overflow-y-auto p-3">
        {renderSection('Navigation', primaryNav)}
        {renderSection('Browse', browseItems)}
        {renderSection('Tools', toolsItems, 'standard')}
        {renderSection('Advanced', advancedItems, 'advanced')}
      </div>

      {/* Bottom: Mode Selector + Collapse Toggle */}
      <div className="border-t border-[var(--border)] p-3">
        {!leftSidebarCollapsed && (
          <div className="mb-3">
            <ModeSelector compact />
          </div>
        )}

        {/* Collapse toggle (desktop only) */}
        {!isMobile && (
          <button
            onClick={() => setLeftSidebarCollapsed(!leftSidebarCollapsed)}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-md text-[var(--text-muted)] hover:bg-[var(--surface-elevated)] hover:text-[var(--text)] transition-colors"
            aria-label={leftSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={leftSidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <span className="font-mono" aria-hidden="true">
              {leftSidebarCollapsed ? '»' : '«'}
            </span>
            {!leftSidebarCollapsed && (
              <span className="text-sm">Collapse</span>
            )}
          </button>
        )}
      </div>
    </div>
  );

  // Mobile: Full-screen overlay drawer
  if (isMobile) {
    return (
      <>
        {/* Backdrop */}
        {leftSidebarOpen && (
          <div
            className="fixed inset-0 bg-black/50 z-40"
            onClick={closeLeftSidebar}
          />
        )}

        {/* Drawer */}
        <aside
          aria-label="Main navigation"
          className={`
            fixed top-0 left-0 h-full w-72 bg-[var(--surface)] border-r border-[var(--border)] z-50
            transform transition-transform duration-200 ease-out
            ${leftSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
          `}
          style={{ paddingTop: '48px' }} // Below TopBar
        >
          {sidebarContent}
        </aside>
      </>
    );
  }

  // Desktop: Persistent sidebar
  return (
    <aside
      aria-label="Main navigation"
      className="fixed top-12 left-0 h-[calc(100vh-48px)] bg-[var(--surface)] border-r border-[var(--border)] z-30 transition-all duration-200"
      style={{ width: leftSidebarWidth }}
    >
      {sidebarContent}
    </aside>
  );
}
