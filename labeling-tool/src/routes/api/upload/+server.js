import { json } from '@sveltejs/kit';
import { writeFileSync } from 'fs';
import { getAddedImages, saveAddedImages, getImageMap, getUserImagePath } from '$lib/datastore.js';

function generateId(imageMap) {
	// 9000000번대부터 순차 발급
	const addedData = getAddedImages();
	if (addedData.images.length === 0) return '9000001';
	const maxId = Math.max(...addedData.images.map(img => parseInt(img.id)));
	return String(maxId + 1).padStart(7, '0');
}

export async function POST({ request }) {
	const formData = await request.formData();

	const file         = formData.get('file');
	const manufacturer = formData.get('manufacturer');
	const family       = formData.get('family');
	const variant      = formData.get('variant');
	const split        = formData.get('split');
	const xmin         = parseInt(formData.get('xmin') || '0');
	const ymin         = parseInt(formData.get('ymin') || '0');
	const xmax         = parseInt(formData.get('xmax') || '0');
	const ymax         = parseInt(formData.get('ymax') || '0');

	if (!file || !manufacturer || !family || !variant || !split) {
		return json({ success: false, error: 'Missing required fields' }, { status: 400 });
	}

	const imageMap = getImageMap();
	const newId = generateId(imageMap);

	// 이미지 파일 저장 (원본 데이터셋과 분리된 labeling-tool/data/user_images/ 폴더)
	const buffer = Buffer.from(await file.arrayBuffer());
	const destPath = getUserImagePath(newId);
	writeFileSync(destPath, buffer);

	// added_images.json 업데이트
	const addedData = getAddedImages();
	addedData.images.push({
		id: newId,
		manufacturer,
		family,
		variant,
		split,
		bbox: { xmin, ymin, xmax, ymax },
		addedAt: new Date().toISOString()
	});
	saveAddedImages(addedData); // 내부에서 _imageMap 리셋

	return json({ success: true, id: newId });
}
