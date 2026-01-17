import AuditSessionDetail from './AuditSessionDetail';

// For static export with dynamic routes - allow client-side routing
export const dynamicParams = true;

export async function generateStaticParams() {
  // No static paths - all handled via client-side routing
  return [];
}

export default function AuditSessionPage() {
  return <AuditSessionDetail />;
}
