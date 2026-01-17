import AuditSessionDetail from './AuditSessionDetail';

// Required for static export with dynamic routes
export async function generateStaticParams() {
  return [];
}

export default function AuditSessionPage() {
  return <AuditSessionDetail />;
}
