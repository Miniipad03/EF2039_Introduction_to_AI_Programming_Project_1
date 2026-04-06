import { writable } from 'svelte/store';

/** 클라이언트 측 taxonomy 캐시 — 페이지 이동해도 재요청 안 함 */
export const taxonomyCache = writable(null);

export async function loadTaxonomy(force = false) {
	let current;
	taxonomyCache.subscribe(v => { current = v; })();
	if (current && !force) return current;

	const res = await fetch('/api/taxonomy');
	const data = await res.json();
	taxonomyCache.set(data);
	return data;
}
