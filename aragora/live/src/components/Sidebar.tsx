'use client';

import { useEffect, useRef } from 'react';
import Link from 'next/link';
import { useSidebar } from '@/context/SidebarContext';
import { useAuth } from '@/context/AuthContext';

interface NavItem {
  label: string;
  href: string;
  icon?: string;
  requiresAuth?: boolean;
  adminOnly?: boolean;
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

const mainNavItems: NavItem[] = [
  { label: 'Live Debate', href: '/debate', icon: '!' },
  { label: 'Debates', href: '/debates', icon: '#' },
  { label: 'Gauntlet', href: '/gauntlet', icon: '%' },
  { label: 'Leaderboard', href: '/leaderboard', icon: '^' },
  { label: 'Agents', href: '/agents', icon: '&' },
];

const secondaryNavItems: NavItem[] = [
  { label: 'Workflows', href: '/workflows', icon: '>' },
  { label: 'Connectors', href: '/connectors', icon: '<' },
  { label: 'Templates', href: '/templates', icon: '[' },
  { label: 'Analytics', href: '/analytics', icon: '~' },
  { label: 'Memory', href: '/memory', icon: '=' },
  { label: 'Audit Sessions', href: '/audit', icon: '|' },
  { label: 'Documents', href: '/documents', icon: ']' },
  { label: 'Integrations', href: '/integrations', icon: '<' },
];

const adminNavItems: NavItem[] = [
  { label: 'Admin Dashboard', href: '/admin', icon: '!', adminOnly: true },
  { label: 'User Management', href: '/admin/users', icon: '@', adminOnly: true },
  { label: 'Organizations', href: '/admin/organizations', icon: '#', adminOnly: true },
];

export function Sidebar() {
  const { isOpen, close } = useSidebar();
  const { isAuthenticated, user, logout } = useAuth();
  const sidebarRef = useRef<HTMLDivElement>(null);
  const firstFocusableRef = useRef<HTMLButtonElement>(null);

  const isAdmin = user?.role === 'admin';

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

  const renderNavSection = (title: string, items: NavItem[]) => {
    const filteredItems = items.filter(item => {
      if (item.requiresAuth && !isAuthenticated) return false;
      if (item.adminOnly && !isAdmin) return false;
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
      {/* Backdrop overlay */}
      <div
        className={`fixed inset-0 bg-black/70 backdrop-blur-sm z-40 transition-opacity duration-300 ${
          isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
        aria-hidden="true"
      />

      {/* Sidebar panel */}
      <div
        ref={sidebarRef}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
        className={`fixed top-0 left-0 h-full w-72 bg-background border-r border-acid-green/30 z-50 transform transition-transform duration-300 ease-in-out ${
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

          {/* Main navigation */}
          {renderNavSection('Navigate', mainNavItems)}

          {/* Secondary navigation */}
          {renderNavSection('Tools', secondaryNavItems)}

          {/* Admin section */}
          {isAdmin && renderNavSection('Admin', adminNavItems)}
        </div>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-acid-green/30 bg-background">
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
