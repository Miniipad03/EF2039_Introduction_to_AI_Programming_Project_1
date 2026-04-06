<script>
	import { onMount } from 'svelte';
	import { loadTaxonomy } from '$lib/stores.js';

	let taxonomy = null;
	let manufacturer = '';
	let family = '';
	let variant = '';
	let split = 'train';

	let file = null;
	let previewUrl = '';
	let previewImg;
	let canvas;
	let isDragging = false;
	let dragover = false;

	let bbox = null;
	let drawStart = null;
	let drawCurrent = null;

	let isReference = false;

	let uploading = false;
	let successId = null;
	let error = '';

	onMount(async () => {
		const res = await fetch('/api/taxonomy');
		taxonomy = await res.json();
	});

	$: manufacturers = taxonomy ? taxonomy.manufacturers.map(m => m.name) : [];
	$: families = taxonomy && manufacturer
		? (taxonomy.manufacturers.find(m => m.name === manufacturer)?.families ?? []).map(f => f.name)
		: [];
	$: variants = taxonomy && family
		? (taxonomy.manufacturers.flatMap(m => m.families).find(f => f.name === family)?.variants ?? []).map(v => v.name)
		: [];

	function onManufacturerChange() { family = ''; variant = ''; }
	function onFamilyChange() { variant = ''; }

	function handleFile(f) {
		file = f;
		previewUrl = URL.createObjectURL(f);
		bbox = null;
		drawStart = null;
		drawCurrent = null;
	}

	function onDrop(e) {
		e.preventDefault();
		dragover = false;
		const f = e.dataTransfer?.files[0];
		if (f && f.type.startsWith('image/')) handleFile(f);
	}

	function onFileInput(e) {
		const f = e.target.files[0];
		if (f) handleFile(f);
	}

	// Canvas bbox 드로잉
	function getCanvasCoords(e) {
		const rect = canvas.getBoundingClientRect();
		return { x: e.clientX - rect.left, y: e.clientY - rect.top };
	}

	function onMouseDown(e) {
		drawStart = getCanvasCoords(e);
		drawCurrent = drawStart;
		isDragging = true;
	}

	function onMouseMove(e) {
		if (!isDragging) return;
		drawCurrent = getCanvasCoords(e);
		redrawCanvas();
	}

	function onMouseUp(e) {
		if (!isDragging) return;
		isDragging = false;
		drawCurrent = getCanvasCoords(e);
		redrawCanvas();

		// 디스플레이 좌표 → 이미지 자연 좌표 변환
		const scaleX = previewImg.naturalWidth  / previewImg.offsetWidth;
		const scaleY = previewImg.naturalHeight / previewImg.offsetHeight;
		const x1 = Math.min(drawStart.x, drawCurrent.x);
		const y1 = Math.min(drawStart.y, drawCurrent.y);
		const x2 = Math.max(drawStart.x, drawCurrent.x);
		const y2 = Math.max(drawStart.y, drawCurrent.y);
		bbox = {
			xmin: Math.round(x1 * scaleX),
			ymin: Math.round(y1 * scaleY),
			xmax: Math.round(x2 * scaleX),
			ymax: Math.round(y2 * scaleY)
		};
	}

	function redrawCanvas() {
		if (!canvas || !drawStart || !drawCurrent) return;
		const ctx = canvas.getContext('2d');
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.strokeStyle = '#22c55e';
		ctx.lineWidth = 2;
		const x = Math.min(drawStart.x, drawCurrent.x);
		const y = Math.min(drawStart.y, drawCurrent.y);
		const w = Math.abs(drawCurrent.x - drawStart.x);
		const h = Math.abs(drawCurrent.y - drawStart.y);
		ctx.strokeRect(x, y, w, h);
	}

	function syncCanvas() {
		if (!canvas || !previewImg) return;
		canvas.width  = previewImg.offsetWidth;
		canvas.height = previewImg.offsetHeight;
	}

	function clearBbox() {
		bbox = null;
		drawStart = null;
		drawCurrent = null;
		if (canvas) {
			const ctx = canvas.getContext('2d');
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		}
	}

	async function submit() {
		if (!file) { error = '이미지를 선택해주세요.'; return; }
		if (!manufacturer || !family || !variant) { error = '제조사, Family, Variant를 모두 선택해주세요.'; return; }
		if (!bbox) { error = 'BBox를 그려주세요.'; return; }
		error = '';
		uploading = true;

		const form = new FormData();
		form.append('file', file);
		form.append('manufacturer', manufacturer);
		form.append('family', family);
		form.append('variant', variant);
		form.append('split', split);
		form.append('xmin', String(bbox.xmin));
		form.append('ymin', String(bbox.ymin));
		form.append('xmax', String(bbox.xmax));
		form.append('ymax', String(bbox.ymax));
		form.append('isReference', String(isReference));

		const res = await fetch('/api/upload', { method: 'POST', body: form });
		const data = await res.json();
		uploading = false;

		if (data.success) {
			successId = data.id;
			file = null;
			previewUrl = '';
			bbox = null;
			manufacturer = '';
			family = '';
			variant = '';
			isReference = false;
		} else {
			error = data.error ?? '업로드 실패';
		}
	}
</script>

<h1>이미지 추가</h1>

{#if successId}
	<div class="card" style="border:2px solid #22c55e;text-align:center;padding:32px;margin-bottom:24px">
		<p style="color:#15803d;font-weight:600">이미지 추가 완료! ID: {successId}</p>
		<button class="btn-primary" on:click={() => successId = null} style="margin-top:12px">다음 이미지 추가</button>
	</div>
{/if}

<!-- 드롭존 -->
{#if !file}
	<div
		class="drop-zone"
		class:dragover
		on:dragover|preventDefault={() => dragover = true}
		on:dragleave={() => dragover = false}
		on:drop={onDrop}
		on:click={() => document.getElementById('fileInput').click()}
		on:keydown={(e) => e.key === 'Enter' && document.getElementById('fileInput').click()}
		role="button"
		tabindex="0"
	>
		<div style="font-size:40px">📂</div>
		<p style="font-weight:600;margin:8px 0 4px">이미지를 드래그하거나 클릭해서 선택</p>
		<p style="font-size:12px">JPG, PNG, JPEG 지원</p>
	</div>
	<input id="fileInput" type="file" accept="image/*" style="display:none" on:change={onFileInput} />
{/if}

<!-- 미리보기 + BBox 드로잉 -->
{#if previewUrl}
	<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;align-items:start">
		<div>
			<h2>BBox 드로잉</h2>
			<p style="color:#64748b;font-size:13px">이미지 위에서 드래그하여 항공기 영역을 선택하세요.</p>
			<div class="upload-canvas-wrap">
				<img
					src={previewUrl}
					alt="preview"
					bind:this={previewImg}
					on:load={syncCanvas}
					style="max-width:100%;max-height:500px;display:block"
				/>
				<canvas
					bind:this={canvas}
					on:mousedown={onMouseDown}
					on:mousemove={onMouseMove}
					on:mouseup={onMouseUp}
				></canvas>
			</div>
			{#if bbox}
				<p style="font-size:12px;color:#475569;margin-top:8px">
					BBox: ({bbox.xmin}, {bbox.ymin}) → ({bbox.xmax}, {bbox.ymax})
					<button class="btn-ghost" on:click={clearBbox} style="padding:2px 8px;font-size:11px;margin-left:8px">지우기</button>
				</p>
			{:else}
				<p style="font-size:12px;color:#94a3b8;margin-top:8px">BBox가 그려지지 않았습니다.</p>
			{/if}
			<button class="btn-ghost" on:click={() => { file = null; previewUrl = ''; bbox = null; }} style="margin-top:12px">
				다른 이미지 선택
			</button>
		</div>

		<div>
			<h2>메타데이터</h2>
			<div style="display:flex;flex-direction:column;gap:12px">
				<div>
					<label for="up-mfr" style="font-size:13px;font-weight:500;display:block;margin-bottom:4px">제조사</label>
					<select id="up-mfr" bind:value={manufacturer} on:change={onManufacturerChange} style="width:100%">
						<option value="">선택</option>
						{#each manufacturers as m}<option value={m}>{m}</option>{/each}
					</select>
				</div>
				<div>
					<label for="up-fam" style="font-size:13px;font-weight:500;display:block;margin-bottom:4px">Family</label>
					<select id="up-fam" bind:value={family} on:change={onFamilyChange} disabled={!manufacturer} style="width:100%">
						<option value="">선택</option>
						{#each families as f}<option value={f}>{f}</option>{/each}
					</select>
				</div>
				<div>
					<label for="up-var" style="font-size:13px;font-weight:500;display:block;margin-bottom:4px">Variant</label>
					<select id="up-var" bind:value={variant} on:change={onVariantChange} disabled={!family} style="width:100%">
						<option value="">선택</option>
						{#each variants as v}<option value={v}>{v}</option>{/each}
					</select>
				</div>
				<div>
					<span style="font-size:13px;font-weight:500;display:block;margin-bottom:4px">Split</span>
					<div style="display:flex;gap:16px">
						{#each ['train', 'val', 'test'] as s}
							<label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer">
								<input type="radio" bind:group={split} value={s} /> {s}
							</label>
						{/each}
					</div>
				</div>

				<div>
					<label style="display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer">
						<input type="checkbox" bind:checked={isReference} />
						<span>
							<strong>Reference 이미지</strong>
							<span style="color:#64748b"> — 항공기 도면/일러스트 (실제 사진 아님)</span>
						</span>
					</label>
				</div>

				{#if error}
					<p style="color:#ef4444;font-size:13px">{error}</p>
				{/if}

				<button
					class="btn-primary"
					on:click={submit}
					disabled={uploading}
					style="margin-top:8px"
				>
					{uploading ? '업로드 중...' : '이미지 추가'}
				</button>
			</div>
		</div>
	</div>
{/if}
