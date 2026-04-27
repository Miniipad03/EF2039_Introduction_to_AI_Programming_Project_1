import { json } from '@sveltejs/kit';
import { writeFileSync, unlinkSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';
import { tmpdir } from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));
// __dirname: inference-demo/src/routes/api/infer/
// ../../../../scripts/infer.py = inference-demo/scripts/infer.py
const INFER_SCRIPT = join(__dirname, '..', '..', '..', '..', 'scripts', 'infer.py');

export async function POST({ request }) {
	const formData   = await request.formData();
	const imageFile  = formData.get('image');
	const modelPaths = formData.getAll('modelPath');

	if (!imageFile || modelPaths.length === 0) {
		return json({ error: 'image와 modelPath가 필요합니다' }, { status: 400 });
	}

	const xmin = formData.get('xmin');
	const ymin = formData.get('ymin');
	const xmax = formData.get('xmax');
	const ymax = formData.get('ymax');

	const tmpPath = join(tmpdir(), `infer_${Date.now()}_${Math.random().toString(36).slice(2)}.jpg`);
	writeFileSync(tmpPath, Buffer.from(await imageFile.arrayBuffer()));

	try {
		const args = [INFER_SCRIPT, '--model_path', ...modelPaths, '--image', tmpPath, '--top_k', '5'];
		if (xmin !== null) args.push('--bbox', xmin, ymin, xmax, ymax);

		const result = spawnSync(
			'python',
			args,
			{ encoding: 'utf-8', timeout: 120_000 }
		);

		if (result.error) {
			return json({ error: `프로세스 실행 실패: ${result.error.message}` }, { status: 500 });
		}
		if (result.status !== 0) {
			const msg = result.stderr?.trim() || '추론 실패 (exit code: ' + result.status + ')';
			return json({ error: msg }, { status: 500 });
		}

		return json(JSON.parse(result.stdout));
	} catch (err) {
		return json({ error: err.message }, { status: 500 });
	} finally {
		try { unlinkSync(tmpPath); } catch {}
	}
}
