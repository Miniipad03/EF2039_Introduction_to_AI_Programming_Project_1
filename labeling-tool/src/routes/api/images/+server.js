import { json } from '@sveltejs/kit';
import { queryImages } from '$lib/datastore.js';

export function GET({ url }) {
	const manufacturer  = url.searchParams.get('manufacturer') || '';
	const family        = url.searchParams.get('family') || '';
	const variant       = url.searchParams.get('variant') || '';
	const showExcluded  = url.searchParams.get('showExcluded') === 'true';
	const page          = parseInt(url.searchParams.get('page') || '1');
	const limit         = parseInt(url.searchParams.get('limit') || '50');
	const idsParam      = url.searchParams.get('ids') || '';
	const ids           = idsParam ? idsParam.split(',').map(s => s.trim()).filter(Boolean) : null;

	return json(queryImages({ manufacturer, family, variant, showExcluded, page, limit, ids }));
}
