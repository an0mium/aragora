import AuditSessionDetail from './AuditSessionDetail';

// For static export with optional catch-all
export const dynamicParams = false;

export async function generateStaticParams() {
  // Only generate the base route - client handles the rest
  return [{ id: undefined }];
}

export default function AuditSessionPage() {
  return <AuditSessionDetail />;
}
