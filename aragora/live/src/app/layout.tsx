import type { Metadata, Viewport } from 'next';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { AuthProvider } from '@/context/AuthContext';
import { ToastProvider } from '@/context/ToastContext';
import { SidebarProvider } from '@/context/SidebarContext';
import { ConnectionProvider } from '@/context/ConnectionContext';
import { DebateSelectionProvider } from '@/context/DebateSelectionContext';
import { ProgressiveModeProvider } from '@/context/ProgressiveModeContext';
import { AdaptiveModeProvider } from '@/context/AdaptiveModeContext';
import { Sidebar } from '@/components/Sidebar';
import { TopNavigation } from '@/components/navigation/TopNavigation';
import { ConfigHealthBanner } from '@/components/ConfigHealthBanner';
import { GlobalConnectionStatus } from '@/components/GlobalConnectionStatus';
import { CommandPaletteProvider } from '@/context/CommandPaletteContext';
import { CommandPalette } from '@/components/command-palette';

export const metadata: Metadata = {
  metadataBase: new URL('https://aragora.ai'),
  title: 'ARAGORA // LIVE',
  description: 'Real-time AI red team system - Watch decision stress-tests unfold in public.',
  keywords: ['AI', 'red team', 'stress-test', 'gauntlet', 'decision receipts', 'LLM', 'aragora'],
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
    description: 'Real-time AI red team system',
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
                    <CommandPaletteProvider>
                      <ToastProvider>
                        <ErrorBoundary>
                          <Sidebar />
                          <GlobalConnectionStatus />
                          <CommandPalette />
                          <div id="main-content" tabIndex={-1} className="flex flex-col min-h-screen">
                            <TopNavigation />
                            <main className="flex-1">
                              {children}
                            </main>
                          </div>
                        </ErrorBoundary>
                      </ToastProvider>
                    </CommandPaletteProvider>
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
