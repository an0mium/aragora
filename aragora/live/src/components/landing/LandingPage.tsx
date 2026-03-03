'use client';

import { useEffect, useRef } from 'react';
import { useTheme } from '@/context/ThemeContext';
import { useLayout } from '@/context/LayoutContext';
import { Logo } from '@/components/Logo';
import { LeftSidebar } from '@/components/layout/LeftSidebar';
import { Header } from './Header';
import { HeroSection } from './HeroSection';
import { HowItWorksSection } from './HowItWorksSection';
import { ProblemSection } from './ProblemSection';
import { FeatureShowcase } from './FeatureShowcase';
import { IntegrationsGrid } from './IntegrationsGrid';
import { LiveDemoSection } from './LiveDemoSection';
import { PricingSection } from './PricingSection';
import { Footer } from './Footer';

export function LandingPage() {
  const { theme } = useTheme();
  const { leftSidebarOpen, leftSidebarWidth, toggleLeftSidebar, closeLeftSidebar, isMobile } =
    useLayout();

  // Sidebar starts closed on the landing page.
  // LayoutProvider auto-opens it on desktop; setTimeout ensures our close
  // runs after the provider's initialization effect.
  const closedOnMountRef = useRef(false);
  useEffect(() => {
    if (!closedOnMountRef.current) {
      closedOnMountRef.current = true;
      setTimeout(closeLeftSidebar, 0);
    }
  }, [closeLeftSidebar]);

  return (
    <div
      className="min-h-screen"
      style={{
        backgroundColor: 'var(--bg)',
        color: 'var(--text)',
        fontFamily: 'var(--font-landing)',
      }}
      data-landing-theme={theme}
    >
      {/* Logo mark — fixed upper-left, toggles sidebar */}
      <div className="fixed top-3 left-3 z-[60]">
        <Logo size="lg" pixelSize={32} onClick={toggleLeftSidebar} />
      </div>

      {/* Collapsible sidebar — self-hides when closed */}
      <LeftSidebar />

      {/* Main content — shifts right when sidebar is open on desktop */}
      <div
        className="transition-all duration-200"
        style={{ marginLeft: !isMobile && leftSidebarOpen ? leftSidebarWidth : 0 }}
      >
        <Header />
        <HeroSection />
        <HowItWorksSection />
        <ProblemSection />
        <FeatureShowcase />
        <IntegrationsGrid />
        <LiveDemoSection />
        <PricingSection />
        <Footer />
      </div>
    </div>
  );
}
