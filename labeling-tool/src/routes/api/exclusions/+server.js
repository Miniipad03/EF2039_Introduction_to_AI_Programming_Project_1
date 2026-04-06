import { json } from '@sveltejs/kit';
import { getExcluded, saveExcluded } from '$lib/datastore.js';

export function GET() {
	return json(getExcluded());
}

// POST: 이미지 제외 추가
// body: { ids: string[], reason: string }
export async function POST({ request }) {
	const { ids, reason } = await request.json();
	const data = getExcluded();

	for (const id of ids) {
		if (!data.excluded.includes(id)) {
			data.excluded.push(id);
		}
		data.reasons[id] = reason ?? 'unknown';
	}

	saveExcluded(data);
	return json({ success: true, total: data.excluded.length });
}

// DELETE: 이미지 복구 (제외 해제)
// body: { ids: string[] }
export async function DELETE({ request }) {
	const { ids } = await request.json();
	const data = getExcluded();

	data.excluded = data.excluded.filter(id => !ids.includes(id));
	for (const id of ids) {
		delete data.reasons[id];
	}

	saveExcluded(data);
	return json({ success: true, total: data.excluded.length });
}
