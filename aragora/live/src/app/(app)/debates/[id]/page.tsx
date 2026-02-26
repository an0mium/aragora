import DebateDetailClient from './DebateDetailClient';

// Allow runtime debate IDs in standalone/server mode.
// Static export still uses the fallback static param below.
export const dynamicParams = true;

export async function generateStaticParams() {
  return [{ id: '_' }];
}

export default function DebateDetailPage() {
  return <DebateDetailClient />;
}
