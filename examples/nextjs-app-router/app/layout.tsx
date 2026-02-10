import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Aragora - Multi-Agent Debate Platform',
  description: 'Orchestrate AI debates for defensible decisions',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <nav className="navbar">
          <a href="/" className="logo">Aragora</a>
          <div className="nav-links">
            <a href="/debates">Debates</a>
            <a href="/debates/new">New Debate</a>
          </div>
        </nav>
        <main className="container">
          {children}
        </main>
      </body>
    </html>
  );
}
