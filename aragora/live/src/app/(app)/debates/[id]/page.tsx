import DebateDetailClient from './DebateDetailClient';

// Static export requires dynamicParams=false.
// Runtime/standalone mode can enable dynamic IDs.
export const dynamicParams = process.env.NEXT_OUTPUT === 'export' ? false : true;

export async function generateStaticParams() {
  return [{ id: '_' }];
}

export default function DebateDetailPage() {
  return <DebateDetailClient />;
}
