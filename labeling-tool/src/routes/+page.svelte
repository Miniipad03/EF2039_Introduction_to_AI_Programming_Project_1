<script>
	import { onMount } from 'svelte';
	import { loadTaxonomy } from '$lib/stores.js';

	let data = null;
	let loading = true;
	let familySortKey = 'name';
	let familySortAsc = true;
	let variantSortKey = 'name';
	let variantSortAsc = true;

	async function load(force = false) {
		loading = true;
		data = await loadTaxonomy(force);
		loading = false;
	}

	onMount(() => load());

	// family 플랫 리스트
	$: familyRows = data
		? data.manufacturers.flatMap(m =>
			m.families.map(f => ({
				manufacturer: m.name,
				family: f.name,
				train: f.counts.train,
				val: f.counts.val,
				test: f.counts.test,
				total: f.counts.total
			}))
		  )
		: [];

	// variant 플랫 리스트
	$: variantRows = data
		? data.manufacturers.flatMap(m =>
			m.families.flatMap(f =>
				f.variants.map(v => ({
					manufacturer: m.name,
					family: f.name,
					variant: v.name,
					train: v.counts.train,
					val: v.counts.val,
					test: v.counts.test,
					total: v.counts.total
				}))
			)
		  )
		: [];

	function sortedRows(rows, key, asc) {
		return [...rows].sort((a, b) => {
			const av = a[key], bv = b[key];
			if (typeof av === 'number') return asc ? av - bv : bv - av;
			return asc ? av.localeCompare(bv) : bv.localeCompare(av);
		});
	}

	function toggleSort(which, key) {
		if (which === 'family') {
			if (familySortKey === key) familySortAsc = !familySortAsc;
			else { familySortKey = key; familySortAsc = true; }
		} else {
			if (variantSortKey === key) variantSortAsc = !variantSortAsc;
			else { variantSortKey = key; variantSortAsc = true; }
		}
	}

	function sortIcon(which, key) {
		const current = which === 'family' ? familySortKey : variantSortKey;
		const asc     = which === 'family' ? familySortAsc : variantSortAsc;
		if (current !== key) return '';
		return asc ? ' ▲' : ' ▼';
	}

	$: totalImages = data
		? data.manufacturers.reduce((s, m) => s + m.counts.total, 0)
		: 0;
</script>

<h1>통계 대시보드</h1>

{#if loading}
	<div style="display:flex;justify-content:center;align-items:center;height:200px;gap:12px;color:#64748b">
		<div class="spinner"></div>
		<span>데이터 로딩 중...</span>
	</div>
{:else if data}
	<div class="summary-box">
		<div class="summary-item">
			<div class="value">{totalImages.toLocaleString()}</div>
			<div class="label">유효 이미지</div>
		</div>
		<div class="summary-item">
			<div class="value" style="color:#ef4444">{data.excludedTotal}</div>
			<div class="label">제외된 이미지</div>
		</div>
		<div class="summary-item">
			<div class="value">{data.manufacturers.length}</div>
			<div class="label">제조사</div>
		</div>
		<div class="summary-item">
			<div class="value">{familyRows.length}</div>
			<div class="label">Family</div>
		</div>
		<div class="summary-item">
			<div class="value">{variantRows.length}</div>
			<div class="label">Variant</div>
		</div>
		<button class="btn-ghost" on:click={() => load(true)} style="margin-left:auto">↻ 새로고침</button>
	</div>

	<!-- Family 테이블 -->
	<h2>Family별 이미지 수</h2>
	<div style="overflow-x:auto; margin-bottom:32px">
		<table>
			<thead>
				<tr>
					<th on:click={() => toggleSort('family','manufacturer')}>Manufacturer{sortIcon('family','manufacturer')}</th>
					<th on:click={() => toggleSort('family','family')}>Family{sortIcon('family','family')}</th>
					<th on:click={() => toggleSort('family','train')}>Train{sortIcon('family','train')}</th>
					<th on:click={() => toggleSort('family','val')}>Val{sortIcon('family','val')}</th>
					<th on:click={() => toggleSort('family','test')}>Test{sortIcon('family','test')}</th>
					<th on:click={() => toggleSort('family','total')}>Total{sortIcon('family','total')}</th>
				</tr>
			</thead>
			<tbody>
				{#each sortedRows(familyRows, familySortKey, familySortAsc) as row}
					<tr>
						<td>{row.manufacturer}</td>
						<td>{row.family}</td>
						<td><span class="badge badge-train">{row.train}</span></td>
						<td><span class="badge badge-val">{row.val}</span></td>
						<td><span class="badge badge-test">{row.test}</span></td>
						<td><strong>{row.total}</strong></td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>

	<!-- Variant 테이블 -->
	<h2>Variant별 이미지 수</h2>
	<div style="overflow-x:auto">
		<table>
			<thead>
				<tr>
					<th on:click={() => toggleSort('variant','manufacturer')}>Manufacturer{sortIcon('variant','manufacturer')}</th>
					<th on:click={() => toggleSort('variant','family')}>Family{sortIcon('variant','family')}</th>
					<th on:click={() => toggleSort('variant','variant')}>Variant{sortIcon('variant','variant')}</th>
					<th on:click={() => toggleSort('variant','train')}>Train{sortIcon('variant','train')}</th>
					<th on:click={() => toggleSort('variant','val')}>Val{sortIcon('variant','val')}</th>
					<th on:click={() => toggleSort('variant','test')}>Test{sortIcon('variant','test')}</th>
					<th on:click={() => toggleSort('variant','total')}>Total{sortIcon('variant','total')}</th>
				</tr>
			</thead>
			<tbody>
				{#each sortedRows(variantRows, variantSortKey, variantSortAsc) as row}
					<tr>
						<td>{row.manufacturer}</td>
						<td>{row.family}</td>
						<td>{row.variant}</td>
						<td><span class="badge badge-train">{row.train}</span></td>
						<td><span class="badge badge-val">{row.val}</span></td>
						<td><span class="badge badge-test">{row.test}</span></td>
						<td><strong>{row.total}</strong></td>
					</tr>
				{/each}
			</tbody>
		</table>
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
