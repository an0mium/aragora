'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

interface NavItem {
  label: string;
  href: string;
  icon: string;
}

const navItems: NavItem[] = [
  { label: 'Overview', href: '/admin', icon: '~' },
  { label: 'Users', href: '/admin/users', icon: '@' },
  { label: 'Organizations', href: '/admin/organizations', icon: '#' },
  { label: 'Billing', href: '/admin/billing', icon: '$' },
  { label: 'Audit Logs', href: '/admin/audit', icon: '!' },
  { label: 'Security', href: '/admin/security', icon: '*' },
  { label: 'Revenue', href: '/admin/revenue', icon: '%' },
  { label: 'Queue', href: '/admin/queue', icon: '>' },
  { label: 'Training', href: '/admin/training', icon: '^' },
  { label: 'Personas', href: '/admin/personas', icon: '&' },
];

interface AdminSidebarProps {
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

export function AdminSidebar({ collapsed = false, onToggleCollapse }: AdminSidebarProps) {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === '/admin') {
      return pathname === '/admin';
    }
    return pathname?.startsWith(href);
  };

  return (
    <aside
      className={`
        fixed top-12 left-0 h-[calc(100vh-48px)] bg-surface border-r border-acid-green/30 z-30
        transition-all duration-200 ease-out
        ${collapsed ? 'w-16' : 'w-56'}
      `}
    >
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-3 border-b border-acid-green/20">
          {!collapsed && (
            <div className="font-mono text-sm text-acid-green">
              ADMIN PANEL
            </div>
          )}
          {collapsed && (
            <div className="font-mono text-lg text-acid-green text-center">
              A
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-2 space-y-1">
          {navItems.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`
                  flex items-center gap-3 px-3 py-2 rounded-md transition-colors font-mono text-sm
                  ${active
                    ? 'bg-acid-green/10 text-acid-green'
                    : 'text-text-muted hover:bg-surface-elevated hover:text-text'
                  }
                `}
                title={collapsed ? item.label : undefined}
              >
                <span className="w-5 text-center">{item.icon}</span>
                {!collapsed && <span>{item.label}</span>}
              </Link>
            );
          })}
        </nav>

        {/* Collapse Toggle */}
        <div className="p-3 border-t border-acid-green/20">
          <button
            onClick={onToggleCollapse}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-md text-text-muted hover:bg-surface-elevated hover:text-text transition-colors font-mono text-sm"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            <span>{collapsed ? '>>' : '<<'}</span>
            {!collapsed && <span>Collapse</span>}
          </button>
        </div>
      </div>
    </aside>
  );
}

export default AdminSidebar;
