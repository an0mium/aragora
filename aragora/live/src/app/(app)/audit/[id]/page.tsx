import AuditSessionDetail from './AuditSessionDetail';

// For static export, we need generateStaticParams
// Since audit IDs are dynamic, we return empty array (client-side routing handles this)
export function generateStaticParams() {
  return [];
}

export default function AuditSessionPage() {
  return <AuditSessionDetail />;
}
