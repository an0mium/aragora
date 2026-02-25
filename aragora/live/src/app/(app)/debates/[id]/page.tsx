import DebateDetailClient from './DebateDetailClient';

// For static export - no pages pre-rendered, client-side navigation only
export const dynamicParams = false;

export async function generateStaticParams() {
  return [{ id: '_' }];
}

export default function DebateDetailPage() {
  return <DebateDetailClient />;
}
