import type { PageServerLoad } from './$types';
import { getServerClient } from '$lib/aragora';
import { error } from '@sveltejs/kit';

export const load: PageServerLoad = async ({ params }) => {
  const client = getServerClient();

  try {
    const debate = await client.debates.get(params.id);
    return { debate };
  } catch {
    throw error(404, 'Debate not found');
  }
};
