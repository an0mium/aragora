import type { PageServerLoad } from './$types';
import { getServerClient } from '$lib/aragora';

export const load: PageServerLoad = async () => {
  const client = getServerClient();

  try {
    const response = await client.debates.list({ limit: 20 });
    return {
      debates: response.debates || [],
    };
  } catch (error) {
    console.error('Failed to fetch debates:', error);
    return {
      debates: [],
    };
  }
};
