import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  Link,
} from '@remix-run/react';
import type { LinksFunction } from '@remix-run/node';

import styles from './root.css?url';

export const links: LinksFunction = () => [{ rel: 'stylesheet', href: styles }];

export default function App() {
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        <nav className="navbar">
          <Link to="/" className="logo">Aragora</Link>
          <div className="nav-links">
            <Link to="/debates">Debates</Link>
            <Link to="/debates/new">New Debate</Link>
          </div>
        </nav>
        <main className="container">
          <Outlet />
        </main>
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}
