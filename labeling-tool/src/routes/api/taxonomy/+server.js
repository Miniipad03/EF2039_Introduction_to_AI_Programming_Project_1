import { json } from '@sveltejs/kit';
import { getTaxonomyWithCounts } from '$lib/datastore.js';

export function GET() {
	return json(getTaxonomyWithCounts());
}
