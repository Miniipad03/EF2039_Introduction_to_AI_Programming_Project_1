<script>
	import { onMount } from 'svelte';
	import { loadTaxonomy } from '$lib/stores.js';

	// 필터 상태
	let manufacturer = '';
	let family = '';
	let variant = '';
	let idSearch = '';   // 쉼표 구분 ID 검색
	let page = 1;
	let limit = 50;

	// 데이터
	let taxonomy = null;
	let result = { images: [], total: 0, pages: 1 };
	let loading = false;

	// 선택 상태
	let selected = new Set();
	let reason = 'partial_aircraft';

	// bbox 오버레이 표시 여부
	let showBbox = true;

	// 이미지 렌더 메타 (object-fit: contain 기준 pixel 좌표 계산용)
	let imgMeta = {}; // { [id]: { scale, ox, oy } }

	onMount(async () => {
		taxonomy = await loadTaxonomy();
		await loadImages();
	});

	async function loadImages() {
		loading = true;
		selected = new Set();
		imgMeta = {};

		const ids = idSearch.split(',').map(s => s.trim()).filter(Boolean);

		const params = new URLSearchParams({ page, limit });
		if (ids.length > 0) {
			params.set('ids', ids.join(','));
		} else {
			if (manufacturer) params.set('manufacturer', manufacturer);
			if (family)       params.set('family', family);
			if (variant)      params.set('variant', variant);
		}

		const res = await fetch('/api/images?' + params);
		result = await res.json();
		loading = false;
	}

	// 캐스케이딩 드롭다운
	$: manufacturers = taxonomy ? taxonomy.manufacturers.map(m => m.name) : [];
	$: families = taxonomy && manufacturer
		? (taxonomy.manufacturers.find(m => m.name === manufacturer)?.families ?? []).map(f => f.name)
		: taxonomy ? taxonomy.manufacturers.flatMap(m => m.families.map(f => f.name)) : [];
	$: variants = taxonomy && family
		? (taxonomy.manufacturers.flatMap(m => m.families).find(f => f.name === family)?.variants ?? []).map(v => v.name)
		: [];

	function onManufacturerChange() { family = ''; variant = ''; page = 1; loadImages(); }
	function onFamilyChange()        { variant = ''; page = 1; loadImages(); }
	function onVariantChange()       { page = 1; loadImages(); }

	function onIdSearchKeydown(e) {
		if (e.key === 'Enter') { page = 1; loadImages(); }
	}
	function clearIdSearch() { idSearch = ''; page = 1; loadImages(); }

	function toggleSelect(id) {
		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
		selected = new Set(selected);
	}

	async function confirmExclusion() {
		if (selected.size === 0) return;
		await fetch('/api/exclusions', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ ids: [...selected], reason })
		});
		selected = new Set();
		await loadImages();
	}

	function cancelSelection() { selected = new Set(); }

	// 이미지 로드 시 object-fit:contain 기준 렌더 영역 계산
	function onImgLoad(e, id) {
		const img = e.currentTarget;
		const natW = img.naturalWidth;
		const natH = img.naturalHeight;
		const W = img.offsetWidth;
		const H = img.offsetHeight;
		if (!natW || !natH || !W || !H) return;
		const scale = Math.min(W / natW, H / natH);
		const ox = (W - natW * scale) / 2;
		const oy = (H - natH * scale) / 2;
		imgMeta = { ...imgMeta, [id]: { scale, ox, oy } };
	}

	function bboxStyle(bbox, id) {
		const meta = imgMeta[id];
		if (!bbox || !meta) return 'display:none';
		const { scale, ox, oy } = meta;
		const l = (ox + bbox.xmin * scale).toFixed(1);
		const t = (oy + bbox.ymin * scale).toFixed(1);
		const w = ((bbox.xmax - bbox.xmin) * scale).toFixed(1);
		const h = ((bbox.ymax - bbox.ymin) * scale).toFixed(1);
		return `left:${l}px;top:${t}px;width:${w}px;height:${h}px`;
	}
</script>

<h1>이미지 브라우저</h1>

<!-- 필터 -->
<div class="filter-row">
	<select bind:value={manufacturer} on:change={onManufacturerChange} disabled={!!idSearch}>
		<option value="">전체 제조사</option>
		{#each manufacturers as m}<option value={m}>{m}</option>{/each}
	</select>
	<select bind:value={family} on:change={onFamilyChange} disabled={!manufacturer || !!idSearch}>
		<option value="">전체 Family</option>
		{#each families as f}<option value={f}>{f}</option>{/each}
	</select>
	<select bind:value={variant} on:change={onVariantChange} disabled={!family || !!idSearch}>
		<option value="">전체 Variant</option>
		{#each variants as v}<option value={v}>{v}</option>{/each}
	</select>

	<div style="display:flex;align-items:center;gap:4px">
		<input
			type="text"
			placeholder="ID 검색 (쉼표 구분)"
			bind:value={idSearch}
			on:keydown={onIdSearchKeydown}
			style="padding:7px 10px;border:1px solid #d1d5db;border-radius:6px;font-size:13px;width:200px"
		/>
		{#if idSearch}
			<button class="btn-ghost" on:click={clearIdSearch} style="padding:6px 10px;font-size:12px">✕</button>
		{/if}
	</div>

	<label style="display:flex;align-items:center;gap:6px;font-size:13px;color:#475569">
		<input type="checkbox" bind:checked={showBbox} /> BBox 표시
	</label>
	<select bind:value={limit} on:change={() => { page = 1; loadImages(); }}>
		<option value={25}>25개씩</option>
		<option value={50}>50개씩</option>
		<option value={100}>100개씩</option>
	</select>
	<span style="color:#64748b;font-size:13px">{result.total}개 이미지</span>
</div>

<!-- 이미지 그리드 -->
{#if loading}
	<div style="display:flex;justify-content:center;align-items:center;height:200px;gap:12px;color:#64748b">
		<div class="spinner"></div>
		<span>이미지 로딩 중...</span>
	</div>
{:else}
	<div class="grid-images" style="margin-bottom: {selected.size > 0 ? '80px' : '0'}">
		{#each result.images as img (img.id)}
			<div
				class="image-card"
				class:selected={selected.has(img.id)}
				on:click={() => toggleSelect(img.id)}
				on:keydown={(e) => e.key === 'Enter' && toggleSelect(img.id)}
				role="checkbox"
				aria-checked={selected.has(img.id)}
				tabindex="0"
			>
				<div class="img-wrap">
					<img
						src="/api/image/{img.id}"
						alt={img.variant}
						loading="lazy"
						on:load={(e) => onImgLoad(e, img.id)}
					/>
					{#if showBbox && img.bbox}
						<div class="bbox-overlay" style={bboxStyle(img.bbox, img.id)}></div>
					{/if}
				</div>
				<div class="label">{img.id} · {img.variant}</div>
			</div>
		{/each}
	</div>

	<!-- 페이지네이션 -->
	{#if result.pages > 1}
		<div class="pagination">
			<button on:click={() => { page = Math.max(1, page - 1); loadImages(); }} disabled={page === 1}>‹</button>
			{#each Array(result.pages) as _, i}
				{#if Math.abs(i + 1 - page) < 3 || i === 0 || i === result.pages - 1}
					<button class:active={page === i + 1} on:click={() => { page = i + 1; loadImages(); }}>{i + 1}</button>
				{:else if Math.abs(i + 1 - page) === 3}
					<span style="padding:0 4px;color:#94a3b8">…</span>
				{/if}
			{/each}
			<button on:click={() => { page = Math.min(result.pages, page + 1); loadImages(); }} disabled={page === result.pages}>›</button>
		</div>
	{/if}
{/if}

<!-- 하단 액션바 -->
{#if selected.size > 0}
	<div class="action-bar">
		<span>{selected.size}개 선택됨</span>
		<select bind:value={reason} style="background:#334155;color:white;border-color:#475569">
			<option value="partial_aircraft">부분 항공기</option>
			<option value="silhouette">역광/실루엣</option>
			<option value="low_quality">저품질</option>
			<option value="other">기타</option>
		</select>
		<button class="btn-danger" on:click={confirmExclusion}>제외 확정</button>
		<button class="btn-ghost" on:click={cancelSelection}>취소</button>
	</div>
{/if}

<style>
	.spinner {
		width: 20px;
		height: 20px;
		border: 2px solid #e2e8f0;
		border-top-color: #3b82f6;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }
</style>
