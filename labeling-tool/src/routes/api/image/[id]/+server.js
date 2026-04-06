import { readFileSync, existsSync } from 'fs';
import { getImagePath } from '$lib/datastore.js';

export function GET({ params }) {
	const { id } = params;

	// id는 파일명만 허용 (경로 탐색 방지)
	if (!/^\d{7}$/.test(id)) {
		return new Response('Invalid image ID', { status: 400 });
	}

	const filePath = getImagePath(id);

	if (!existsSync(filePath)) {
		return new Response('Image not found', { status: 404 });
	}

	const buffer = readFileSync(filePath);
	return new Response(buffer, {
		headers: {
			'Content-Type': 'image/jpeg',
			'Cache-Control': 'public, max-age=3600'
		}
	});
}
