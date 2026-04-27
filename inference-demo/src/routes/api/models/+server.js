import { json } from '@sveltejs/kit';
import { readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
// __dirname: inference-demo/src/routes/api/models/
// ../../../../../ = EF2039_Introduction_to_AI_Programming_Project_1/
const PROJECT_ROOT = join(__dirname, '..', '..', '..', '..', '..');

function parseModelMeta(filename) {
	const stem = filename.replace(/\.pth$/, '');
	const m = stem.match(/^best_fold(\d+)_(.+)$/);
	if (!m) return { fold: null, model: stem, attn: 'noattn', data_tag: '', label: '' };

	const fold = parseInt(m[1]);
	const tag  = m[2].split('_lr')[0];  // drop _lr... suffix
	const parts = tag.split('_');

	return {
		fold,
		model:    parts[0] ?? 'resnet34',
		attn:     parts[1] ?? 'noattn',
		data_tag: parts[2] ?? '',
		label:    parts[3] ?? 'family',
	};
}

function findPthFiles(dir) {
	const results = [];

	function scan(current) {
		let entries;
		try { entries = readdirSync(current, { withFileTypes: true }); }
		catch { return; }

		for (const entry of entries) {
			if (entry.name.startsWith('.') || entry.name === 'node_modules') continue;
			const full = join(current, entry.name);
			if (entry.isDirectory()) {
				scan(full);
			} else if (entry.name.endsWith('.pth')) {
				results.push({
					path: full,
					name: entry.name,
					meta: parseModelMeta(entry.name),
				});
			}
		}
	}

	scan(dir);
	return results;
}

export function GET() {
	const files = findPthFiles(PROJECT_ROOT);
	files.sort((a, b) => a.name.localeCompare(b.name));
	return json(files);
}
