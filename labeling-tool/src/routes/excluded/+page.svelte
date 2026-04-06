<script>
	import { onMount } from 'svelte';

	let result = { images: [], total: 0, pages: 1 };
	let loading = false;
	let page = 1;
	let limit = 50;
	let selected = new Set();

	onMount(() => loadImages());

	async function loadImages() {
		loading = true;
		selected = new Set();
		const params = new URLSearchParams({ showExcluded: 'true', page, limit });
		const res = await fetch('/api/images?' + params);
		result = await res.json();
		loading = false;
	}

	function toggleSelect(id) {
		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
		selected = new Set(selected);
	}

	async function restoreSelected() {
		if (selected.size === 0) return;
		await fetch('/api/exclusions', {
			method: 'DELETE',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ ids: [...selected] })
		});
		selected = new Set();
		await loadImages();
	}

	const reasonLabels = {
		partial_aircraft: '부분 항공기',
		silhouette: '역광/실루엣',
		low_quality: '저품질',
		other: '기타'
	};
</script>

<h1>제외된 이미지 <span style="font-size:16px;color:#ef4444">({result.total}개)</span></h1>

{#if loading}
	<p>Loading...</p>
{:else if result.total === 0}
	<div class="card" style="text-align:center;padding:48px;color:#64748b">
		<p>제외된 이미지가 없습니다.</p>
		<a href="/browse" style="color:#3b82f6">이미지 브라우저로 이동</a>
	</div>
{:else}
	<div class="grid-images" style="margin-bottom: {selected.size > 0 ? '80px' : '0'}">
		{#each result.images as img (img.id)}
			<div
				class="image-card excluded-card"
				class:selected={selected.has(img.id)}
				on:click={() => toggleSelect(img.id)}
				on:keydown={(e) => e.key === 'Enter' && toggleSelect(img.id)}
				role="checkbox"
				aria-checked={selected.has(img.id)}
				tabindex="0"
			>
				<div class="img-wrap">
					<img src="/api/image/{img.id}" alt={img.variant} loading="lazy" />
					<div class="excluded-banner">{reasonLabels[img.reason] ?? img.reason ?? 'EXCLUDED'}</div>
				</div>
				<div class="label">{img.id} · {img.variant}</div>
			</div>
		{/each}
	</div>

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

{#if selected.size > 0}
	<div class="action-bar">
		<span>{selected.size}개 선택됨</span>
		<button class="btn-success" on:click={restoreSelected}>복구</button>
		<button class="btn-ghost" on:click={() => selected = new Set()}>취소</button>
	</div>
{/if}
