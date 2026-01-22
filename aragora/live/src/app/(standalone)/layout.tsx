export default function StandaloneLayout({ children }: { children: React.ReactNode }) {
  // No AppShell - pages in this group provide their own headers/navigation
  return <>{children}</>;
}
