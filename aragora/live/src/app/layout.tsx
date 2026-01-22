import type { Metadata, Viewport } from 'next';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { AuthProvider } from '@/context/AuthContext';
import { ToastProvider } from '@/context/ToastContext';
import { ConnectionProvider } from '@/context/ConnectionContext';
import { DebateSelectionProvider } from '@/context/DebateSelectionContext';
import { ProgressiveModeProvider } from '@/context/ProgressiveModeContext';
import { AdaptiveModeProvider } from '@/context/AdaptiveModeContext';
import { LayoutProvider } from '@/context/LayoutContext';
import { RightSidebarProvider } from '@/context/RightSidebarContext';
import { SidebarProvider } from '@/context/SidebarContext';
import { ConfigHealthBanner } from '@/components/ConfigHealthBanner';
import { CommandPaletteProvider } from '@/context/CommandPaletteContext';
import { CommandPalette } from '@/components/command-palette';

export const metadata: Metadata = {
  metadataBase: new URL('https://aragora.ai'),
  title: 'ARAGORA // LIVE',
  description: 'Omnivorous Multi Agent Decision Making Engine - Ingest from any source, query from any channel, get multi-agent consensus.',
  keywords: ['AI', 'multi-agent', 'decision making', 'debate', 'consensus', 'LLM', 'aragora', 'Claude', 'GPT', 'Gemini', 'Slack', 'Telegram', 'API'],
  authors: [{ name: 'Aragora Team' }],
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Aragora',
  },
  icons: {
    icon: '/icon.png',
    apple: '/apple-icon.png',
  },
  openGraph: {
    title: 'ARAGORA // LIVE',
    description: 'Omnivorous Multi Agent Decision Making Engine - Any source, any channel, multi-agent consensus.',
    type: 'website',
    images: ['/aragora-logo.png'],
  },
};

// Viewport configuration - separate export in Next.js 14+
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  viewportFit: 'cover',
  themeColor: '#00ff41',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="crt-overlay crt-flicker">
        {/* Configuration health banner - shows warnings for missing env vars */}
        <ConfigHealthBanner />
        {/* Skip to main content link for keyboard/screen reader users */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:bg-background focus:text-accent focus:px-4 focus:py-2 focus:border focus:border-accent"
        >
          Skip to main content
        </a>
        <AuthProvider>
          <ConnectionProvider>
            <DebateSelectionProvider>
              <ProgressiveModeProvider>
                <AdaptiveModeProvider>
                  <SidebarProvider>
                    <LayoutProvider>
                      <RightSidebarProvider>
                        <CommandPaletteProvider>
                          <ToastProvider>
                            <ErrorBoundary>
                              <CommandPalette />
                              {children}
                            </ErrorBoundary>
                          </ToastProvider>
                        </CommandPaletteProvider>
                      </RightSidebarProvider>
                    </LayoutProvider>
                  </SidebarProvider>
                </AdaptiveModeProvider>
              </ProgressiveModeProvider>
            </DebateSelectionProvider>
          </ConnectionProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
