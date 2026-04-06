import { readFileSync, writeFileSync, readdirSync, unlinkSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

// 데이터 경로: src/lib/ → (..×2) → labeling-tool/ → (..) → Project_1/
const DATA_ROOT = join(__dirname, '..', '..', '..', 'data', 'fgvc-aircraft-2013b', 'data');
// labeling-tool/data/: src/lib/ → (..×2) → labeling-tool/data/
const TOOL_DATA = join(__dirname, '..', '..', 'data');
// 사용자가 추가한 이미지 저장 폴더 (원본 데이터셋과 분리)
const USER_IMAGES  = join(TOOL_DATA, 'user_images');
// 파일 분리 방식: ID별 개별 파일 → 팀원 동시 작업 시 git 충돌 방지
const EXCLUDED_DIR  = join(TOOL_DATA, 'excluded');
const ADDED_DIR     = join(TOOL_DATA, 'added_images');

// ─── 파일 파싱 유틸 ───────────────────────────────────────────

function readLines(filePath) {
	return readFileSync(filePath, 'utf-8').split('\n').filter(Boolean);
}

/**
 * "image_id label name with spaces" 형식 파싱
 * 반환: Map<imageId, label>
 */
function parseAnnotationFile(filePath) {
	const map = new Map();
	for (const line of readLines(filePath)) {
		const spaceIdx = line.indexOf(' ');
		const id = line.slice(0, spaceIdx).trim();
		const label = line.slice(spaceIdx + 1).trim();
		map.set(id, label);
	}
	return map;
}

/**
 * images_box.txt 파싱
 * 반환: Map<imageId, {xmin, ymin, xmax, ymax}>
 */
function parseBboxFile(filePath) {
	const map = new Map();
	for (const line of readLines(filePath)) {
		const parts = line.trim().split(/\s+/);
		if (parts.length < 5) continue;
		map.set(parts[0], {
			xmin: parseInt(parts[1]),
			ymin: parseInt(parts[2]),
			xmax: parseInt(parts[3]),
			ymax: parseInt(parts[4])
		});
	}
	return map;
}

// ─── 계층 구조 + 이미지 맵 구축 (서버 시작 시 1회) ───────────────
// globalThis에 캐시 → dev 모드 HMR로 모듈이 재로드돼도 데이터 유지

/** @returns {{ imageMap: Map, bboxMap: Map, hierarchy: object } | null} */
function getCache() { return globalThis.__fgvcCache ?? null; }
function setCache(data) { globalThis.__fgvcCache = data; }
function clearCache() { globalThis.__fgvcCache = null; }

function buildData() {
	if (getCache()) return;

	// 각 split별 annotation 파일 로드
	const splits = ['train', 'val', 'test'];
	const mfr = {}, fam = {}, vari = {};

	for (const split of splits) {
		mfr[split]  = parseAnnotationFile(join(DATA_ROOT, `images_manufacturer_${split}.txt`));
		fam[split]  = parseAnnotationFile(join(DATA_ROOT, `images_family_${split}.txt`));
		vari[split] = parseAnnotationFile(join(DATA_ROOT, `images_variant_${split}.txt`));
	}

	const bboxMap = parseBboxFile(join(DATA_ROOT, 'images_box.txt'));

	// imageMap 구축
	const imageMap = new Map();
	for (const split of splits) {
		for (const [id, manufacturer] of mfr[split]) {
			imageMap.set(id, {
				manufacturer,
				family:  fam[split].get(id) ?? '',
				variant: vari[split].get(id) ?? '',
				split
			});
		}
	}

	// added_images.json에서 추가된 이미지 병합
	const added = getAddedImages();
	for (const img of added.images) {
		imageMap.set(img.id, {
			manufacturer: img.manufacturer,
			family:       img.family,
			variant:      img.variant,
			split:        img.split,
			isAdded:      true
		});
		if (img.bbox) {
			bboxMap.set(img.id, img.bbox);
		}
	}

	// 계층 구조 구축
	const mfrMap = new Map();
	for (const { manufacturer, family, variant } of imageMap.values()) {
		if (!mfrMap.has(manufacturer)) mfrMap.set(manufacturer, new Map());
		const famMap = mfrMap.get(manufacturer);
		if (!famMap.has(family)) famMap.set(family, new Set());
		famMap.get(family).add(variant);
	}

	const hierarchy = {
		manufacturers: [...mfrMap.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([mName, famMap]) => ({
			name: mName,
			families: [...famMap.entries()].sort((a, b) => a[0].localeCompare(b[0])).map(([fName, varSet]) => ({
				name: fName,
				variants: [...varSet].sort()
			}))
		}))
	};

	setCache({ imageMap, bboxMap, hierarchy });
}

// ─── Public API ───────────────────────────────────────────────

/** 전체 이미지 맵 반환 */
export function getImageMap() {
	buildData();
	return getCache().imageMap;
}

/** bbox 맵 반환 */
export function getBboxMap() {
	buildData();
	return getCache().bboxMap;
}

/** 계층 구조 반환 */
export function getHierarchy() {
	buildData();
	return getCache().hierarchy;
}

/** excluded/ 디렉토리 전체 읽기 → { excluded: string[], reasons: Record<string,string> } */
export function getExcluded() {
	const files = readdirSync(EXCLUDED_DIR).filter(f => f.endsWith('.json'));
	const excluded = [];
	const reasons = {};
	for (const file of files) {
		const data = JSON.parse(readFileSync(join(EXCLUDED_DIR, file), 'utf-8'));
		excluded.push(data.id);
		reasons[data.id] = data.reason ?? 'unknown';
	}
	return { excluded, reasons };
}

/** 이미지 제외 — ID별 파일 저장 (충돌 없음) */
export function excludeImages(ids, reason = 'unknown') {
	for (const id of ids) {
		writeFileSync(
			join(EXCLUDED_DIR, `${id}.json`),
			JSON.stringify({ id, reason, excludedAt: new Date().toISOString() }, null, 2),
			'utf-8'
		);
	}
	clearCache();
}

/** 제외 해제 — 파일 삭제 */
export function unexcludeImages(ids) {
	for (const id of ids) {
		const p = join(EXCLUDED_DIR, `${id}.json`);
		if (existsSync(p)) unlinkSync(p);
	}
	clearCache();
}

/** added_images/ 디렉토리 전체 읽기 → { images: object[] } */
export function getAddedImages() {
	const files = readdirSync(ADDED_DIR).filter(f => f.endsWith('.json'));
	const images = files.map(f => JSON.parse(readFileSync(join(ADDED_DIR, f), 'utf-8')));
	return { images };
}

/** 추가 이미지 1개 저장 — ID별 파일 (충돌 없음) */
export function saveAddedImage(imageData) {
	writeFileSync(
		join(ADDED_DIR, `${imageData.id}.json`),
		JSON.stringify(imageData, null, 2),
		'utf-8'
	);
	clearCache();
}

/** 이미지 파일 경로 반환 */
export function getImagePath(id) {
	buildData();
	const info = getCache().imageMap.get(id);
	// 사용자가 추가한 이미지는 원본 데이터셋과 분리된 폴더에서 읽음
	if (info?.isAdded) {
		return join(USER_IMAGES, `${id}.jpg`);
	}
	return join(DATA_ROOT, 'images', `${id}.jpg`);
}

/** 사용자 추가 이미지 저장 경로 반환 */
export function getUserImagePath(id) {
	return join(USER_IMAGES, `${id}.jpg`);
}

/**
 * 필터링된 이미지 목록 반환
 * @param {Object} opts - {manufacturer, family, variant, showExcluded, page, limit}
 */
export function queryImages({ manufacturer, family, variant, showExcluded = false, page = 1, limit = 50, ids = null }) {
	buildData();
	const excluded = getExcluded();
	const excludedSet = new Set(excluded.excluded);

	let entries = [...getCache().imageMap.entries()];

	// 필터링
	if (ids && ids.length > 0) {
		const idSet = new Set(ids);
		entries = entries.filter(([id]) => idSet.has(id));
	} else {
		if (manufacturer) entries = entries.filter(([, v]) => v.manufacturer === manufacturer);
		if (family)       entries = entries.filter(([, v]) => v.family === family);
		if (variant)      entries = entries.filter(([, v]) => v.variant === variant);
		if (!showExcluded) entries = entries.filter(([id]) => !excludedSet.has(id));
		else               entries = entries.filter(([id]) => excludedSet.has(id));
	}

	const total = entries.length;
	const pages = Math.ceil(total / limit);
	const slice = entries.slice((page - 1) * limit, page * limit);

	return {
		images: slice.map(([id, info]) => ({
			id,
			...info,
			excluded: excludedSet.has(id),
			reason: excluded.reasons[id] ?? null,
			bbox: getCache().bboxMap.get(id) ?? null
		})),
		total,
		page,
		pages
	};
}

/**
 * 계층별 카운트 (exclusion 적용)
 * 반환: hierarchy에 counts 필드 추가된 구조
 */
export function getTaxonomyWithCounts() {
	buildData();
	const excluded = getExcluded();
	const excludedSet = new Set(excluded.excluded);

	// 집계: (manufacturer, family, variant, split) 별 카운트
	const counts = {}; // key: `${manufacturer}|||${family}|||${variant}|||${split}` -> number

	for (const [id, info] of getCache().imageMap.entries()) {
		if (excludedSet.has(id)) continue;
		const key = `${info.manufacturer}|||${info.family}|||${info.variant}|||${info.split}`;
		counts[key] = (counts[key] ?? 0) + 1;
	}

	function getCount(manufacturer, family, variant, split) {
		const key = `${manufacturer}|||${family}|||${variant}|||${split}`;
		return counts[key] ?? 0;
	}

	function sumCounts(manufacturer, family, variant) {
		const result = { train: 0, val: 0, test: 0 };
		for (const split of ['train', 'val', 'test']) {
			result[split] = getCount(manufacturer, family, variant, split);
		}
		result.total = result.train + result.val + result.test;
		return result;
	}

	const hierarchy = getHierarchy();
	const excludedTotal = excludedSet.size;

	return {
		excludedTotal,
		manufacturers: hierarchy.manufacturers.map(mfr => {
			const mCounts = { train: 0, val: 0, test: 0, total: 0 };
			const families = mfr.families.map(fam => {
				const fCounts = { train: 0, val: 0, test: 0, total: 0 };
				const variants = fam.variants.map(v => {
					const vCounts = sumCounts(mfr.name, fam.name, v);
					fCounts.train += vCounts.train;
					fCounts.val   += vCounts.val;
					fCounts.test  += vCounts.test;
					fCounts.total += vCounts.total;
					return { name: v, counts: vCounts };
				});
				mCounts.train += fCounts.train;
				mCounts.val   += fCounts.val;
				mCounts.test  += fCounts.test;
				mCounts.total += fCounts.total;
				return { name: fam.name, variants, counts: fCounts };
			});
			return { name: mfr.name, families, counts: mCounts };
		})
	};
}

// 서버 모듈 로드 시 즉시 빌드 → 첫 요청도 캐시 히트
buildData();
