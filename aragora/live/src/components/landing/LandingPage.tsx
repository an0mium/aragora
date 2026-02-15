'use client';

import { Header } from './Header';
import { HeroSection } from './HeroSection';
import { VerticalCards } from './VerticalCards';
import { WhyAragoraSection } from './WhyAragoraSection';
import { DebateProtocolSection } from './DebateProtocolSection';
import { TemplatePicker } from '../templates/TemplatePicker';
import { CapabilitiesSection } from './CapabilitiesSection';
import { TrustSection } from './TrustSection';
import { Footer } from './Footer';

export function LandingPage() {
  return (
    <>
      <main className="min-h-screen bg-bg text-text relative z-10 flex flex-col">
        <Header />
        <HeroSection />
        <VerticalCards />
        <WhyAragoraSection />
        <DebateProtocolSection />
        <TemplatePicker compact compactLimit={2} />
        <CapabilitiesSection />
        <TrustSection />
        <Footer />
      </main>
    </>
  );
}
