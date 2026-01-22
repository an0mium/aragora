import { Metadata } from 'next';
import { AgentProfileWrapper } from './AgentProfileWrapper';

// For static export with optional catch-all
export const dynamicParams = false;

export async function generateStaticParams() {
  // Only generate the base route - client handles the rest
  return [{ name: undefined }];
}

export const metadata: Metadata = {
  title: 'Agent Profile | ARAGORA',
  description: 'View detailed agent profiles, statistics, and head-to-head comparisons',
};

export default function AgentProfilePage() {
  return <AgentProfileWrapper />;
}
