import { Metadata } from 'next';
import { AgentProfileWrapper } from './AgentProfileWrapper';

// Allow runtime agent names while still providing a fallback static export path.
export const dynamicParams = true;

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
