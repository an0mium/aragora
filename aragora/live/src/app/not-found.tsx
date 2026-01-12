import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-bg flex items-center justify-center p-4">
      <div className="max-w-2xl w-full border border-warning bg-surface p-6 font-mono">
        <div className="flex items-start gap-3 mb-4">
          <div className="text-warning text-2xl glow-text-subtle">{'>'}</div>
          <div>
            <div className="text-warning font-bold mb-2 text-xl">
              404 - PAGE NOT FOUND
            </div>
            <div className="text-text-muted text-sm mb-2">
              The requested route does not exist in the Aragora network
            </div>
          </div>
        </div>

        <div className="bg-bg border border-border p-4 mb-4 text-text-muted text-sm">
          <div className="mb-2 text-text font-bold">
            {'>'} Available Routes
          </div>
          <ul className="pl-4 space-y-1">
            <li>
              <Link href="/" className="text-accent hover:underline">
                / - Main Dashboard
              </Link>
            </li>
            <li>
              <Link href="/about" className="text-accent hover:underline">
                /about - About Aragora
              </Link>
            </li>
            <li>
              <Link href="/pricing" className="text-accent hover:underline">
                /pricing - Plans & Pricing
              </Link>
            </li>
            <li>
              <Link href="/settings" className="text-accent hover:underline">
                /settings - User Settings
              </Link>
            </li>
          </ul>
        </div>

        <div className="flex gap-3">
          <Link
            href="/"
            className="flex-1 border border-accent text-accent py-2 px-4 hover:bg-accent hover:text-bg transition-colors font-bold text-center"
          >
            {'>'} GO HOME
          </Link>
        </div>

        <div className="mt-6 p-3 bg-surface border border-border text-text-muted text-xs">
          <div className="font-bold mb-1 text-text">{'>'} NEED HELP?</div>
          <ul className="pl-4 space-y-1">
            <li>• Check the URL for typos</li>
            <li>• Use the navigation above</li>
            <li>
              • Contact{' '}
              <a
                href="mailto:support@aragora.ai"
                className="text-accent hover:underline"
              >
                support@aragora.ai
              </a>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
