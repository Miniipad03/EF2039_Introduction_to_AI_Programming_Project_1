import { json } from '@sveltejs/kit';
import { getExcluded, excludeImages, unexcludeImages } from '$lib/datastore.js';

export function GET() {
	return json(getExcluded());
}

// POST: 이미지 제외 추가
// body: { ids: string[], reason: string }
export async function POST({ request }) {
	const { ids, reason } = await request.json();
	excludeImages(ids, reason);
	const data = getExcluded();
	return json({ success: true, total: data.excluded.length });
}

// DELETE: 이미지 복구 (제외 해제)
// body: { ids: string[] }
export async function DELETE({ request }) {
	const { ids } = await request.json();
	unexcludeImages(ids);
	const data = getExcluded();
	return json({ success: true, total: data.excluded.length });
}
