import type { Metadata } from 'next';
import './globals.css';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { AuthProvider } from '@/context/AuthContext';
import { ToastProvider } from '@/context/ToastContext';
import { SidebarProvider } from '@/context/SidebarContext';
import { Sidebar } from '@/components/Sidebar';

export const metadata: Metadata = {
  metadataBase: new URL('https://aragora.ai'),
  title: 'ARAGORA // LIVE',
  description: 'Real-time AI red team system - Watch decision stress-tests unfold in public.',
  keywords: ['AI', 'red team', 'stress-test', 'gauntlet', 'decision receipts', 'LLM', 'aragora'],
  authors: [{ name: 'Aragora Team' }],
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="crt-overlay crt-flicker">
        {/* Skip to main content link for keyboard/screen reader users */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:bg-background focus:text-accent focus:px-4 focus:py-2 focus:border focus:border-accent"
        >
          Skip to main content
        </a>
        <AuthProvider>
          <SidebarProvider>
            <ToastProvider>
              <ErrorBoundary>
                <Sidebar />
                <div id="main-content" tabIndex={-1}>
                  {children}
                </div>
              </ErrorBoundary>
            </ToastProvider>
          </SidebarProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
