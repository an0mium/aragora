import { Metadata } from 'next';
import { GauntletLiveWrapper } from './GauntletLiveWrapper';

// For static export with optional catch-all
export const dynamicParams = false;

export async function generateStaticParams() {
  // Only generate the base route - client handles specific IDs
  return [{ id: undefined }];
}

export const metadata: Metadata = {
  title: 'Live Gauntlet | ARAGORA',
  description: 'Real-time adversarial stress-testing in progress',
};

export default function GauntletLivePage() {
  return <GauntletLiveWrapper />;
}
