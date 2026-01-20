'use client';

import React from 'react';
import { useLayout } from '@/context/LayoutContext';
import { TopBar } from './TopBar';
import { LeftSidebar } from './LeftSidebar';
import { RightSidebar } from './RightSidebar';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps) {
  const {
    leftSidebarOpen,
    leftSidebarCollapsed,
    rightSidebarOpen,
    isMobile,
    leftSidebarWidth,
    rightSidebarWidth,
  } = useLayout();

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--text)]">
      {/* Top Bar - always visible */}
      <TopBar />

      {/* Main layout container */}
      <div className="flex pt-12"> {/* pt-12 = 48px for top bar */}
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Main content area */}
        <main
          className="flex-1 min-w-0 transition-all duration-200"
          style={{
            marginLeft: isMobile ? 0 : leftSidebarOpen ? leftSidebarWidth : 0,
            marginRight: isMobile ? 0 : rightSidebarOpen ? rightSidebarWidth : 0,
          }}
        >
          <div className="h-[calc(100vh-48px)] overflow-auto">
            <div className="max-w-screen-2xl mx-auto px-3 sm:px-4 lg:px-6 py-4">
              {children}
            </div>
          </div>
        </main>

        {/* Right Sidebar */}
        <RightSidebar />
      </div>

      {/* Mobile overlay when left sidebar is open */}
      {isMobile && leftSidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={() => {
            // This will be handled by the context
          }}
        />
      )}
    </div>
  );
}
