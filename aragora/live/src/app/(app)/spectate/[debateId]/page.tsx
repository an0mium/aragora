import SpectateClient from './SpectateClient';

export function generateStaticParams() {
  // Dynamic route - all debate IDs are resolved at runtime via client-side routing
  return [];
}

export default function SpectateDebatePage() {
  return <SpectateClient />;
}
