<script>
	import { onMount } from 'svelte';

	let models = [];
	let selectedModel = '';
	let imageFile = null;
	let imageUrl = '';
	let result = null;
	let inferring = false;
	let errorMsg = '';
	let dragover = false;
	let modelsError = '';

	onMount(async () => {
		try {
			const res = await fetch('/api/models');
			if (!res.ok) { modelsError = `API 오류 (HTTP ${res.status})`; return; }
			models = await res.json();
			if (models.length > 0) selectedModel = models[0].path;
		} catch (e) {
			modelsError = `모델 목록 로드 실패: ${e.message}`;
		}
	});

	$: selectedMeta = models.find(m => m.path === selectedModel)?.meta ?? null;

	function handleFile(f) {
		if (!f || !f.type.startsWith('image/')) return;
		imageFile = f;
		if (imageUrl) URL.revokeObjectURL(imageUrl);
		imageUrl = URL.createObjectURL(f);
		result = null;
		errorMsg = '';
	}

	function onDrop(e) {
		e.preventDefault();
		dragover = false;
		handleFile(e.dataTransfer?.files?.[0]);
	}

	function onFileInput(e) {
		handleFile(e.target.files?.[0]);
		e.target.value = '';
	}

	async function runInference() {
		if (!imageFile || !selectedModel) return;
		inferring = true;
		result = null;
		errorMsg = '';

		const form = new FormData();
		form.append('image', imageFile);
		form.append('modelPath', selectedModel);

		try {
			const res = await fetch('/api/infer', { method: 'POST', body: form });
			const data = await res.json();
			if (!res.ok || data.error) {
				errorMsg = data.error ?? '추론 실패';
			} else {
				result = data;
			}
		} catch (e) {
			errorMsg = e.message;
		} finally {
			inferring = false;
		}
	}

	function pct(conf) {
		return (conf * 100).toFixed(1) + '%';
	}
</script>

<svelte:head><title>FGVC Inference Demo</title></svelte:head>

<div class="demo-layout">

	<!-- ── Left: Controls ── -->
	<div class="col-controls">

		<div class="card">
			<h2>이미지 업로드</h2>
			{#if imageUrl}
				<div>
					<img src={imageUrl} alt="업로드된 이미지" class="preview-img" />
					<button
						class="btn-ghost"
						style="width:100%;margin-top:8px"
						on:click={() => document.getElementById('infer-file').click()}
					>다른 이미지 선택</button>
				</div>
			{:else}
				<div
					class="drop-zone"
					class:dragover
					role="button"
					tabindex="0"
					on:dragover|preventDefault={() => (dragover = true)}
					on:dragleave={() => (dragover = false)}
					on:drop={onDrop}
					on:click={() => document.getElementById('infer-file').click()}
					on:keydown={(e) => e.key === 'Enter' && document.getElementById('infer-file').click()}
				>
					<div style="font-size:36px;margin-bottom:8px">🛩️</div>
					<p style="font-weight:600;margin-bottom:4px">이미지를 드래그하거나 클릭해서 선택</p>
					<p style="font-size:12px">JPG, PNG, WEBP 지원</p>
				</div>
			{/if}
			<input
				id="infer-file"
				type="file"
				accept="image/*"
				style="display:none"
				on:change={onFileInput}
			/>
		</div>

		<div class="card">
			<h2>모델 선택</h2>
			{#if modelsError}
				<p style="color:#ef4444;font-size:13px;line-height:1.6">{modelsError}</p>
			{:else if models.length === 0}
				<p style="color:#94a3b8;font-size:13px;line-height:1.6">
					사용 가능한 .pth 파일이 없습니다.<br/>
					<code>train.py</code>를 실행해 모델을 먼저 학습해주세요.
				</p>
			{:else}
				<select bind:value={selectedModel} style="width:100%">
					{#each models as m}
						<option value={m.path}>{m.name}</option>
					{/each}
				</select>

				{#if selectedMeta}
					<div class="meta-tags">
						<span class="tag tag-model">{selectedMeta.model}</span>
						{#if selectedMeta.attn !== 'noattn'}
							<span class="tag tag-attn">{selectedMeta.attn.toUpperCase()}</span>
						{/if}
						<span class="tag tag-label">{selectedMeta.label}</span>
						<span class="tag tag-data">{selectedMeta.data_tag}</span>
						{#if selectedMeta.fold}
							<span class="tag tag-fold">Fold {selectedMeta.fold}</span>
						{/if}
					</div>
				{/if}
			{/if}
		</div>

		<button
			class="btn-primary btn-run"
			disabled={!imageFile || !selectedModel || inferring}
			on:click={runInference}
		>
			{#if inferring}
				<span class="btn-spinner"></span> 추론 중...
			{:else}
				추론 실행
			{/if}
		</button>
	</div>

	<!-- ── Right: Results ── -->
	<div class="col-results">
		{#if inferring}
			<div class="state-box">
				<div class="spinner-lg"></div>
				<p style="font-weight:600">모델 로딩 및 추론 중...</p>
				<p style="font-size:12px;color:#94a3b8;margin-top:4px">처음 실행 시 수 초가 걸릴 수 있습니다</p>
			</div>

		{:else if errorMsg}
			<div class="card" style="border:1px solid #fecaca">
				<h2 style="color:#ef4444">오류 발생</h2>
				<pre class="error-pre">{errorMsg}</pre>
			</div>

		{:else if result}
			<!-- Top prediction hero -->
			<div class="card hero-card">
				<div class="hero-eyebrow">최상위 예측</div>
				<div class="hero-label">{result.predictions[0].label}</div>
				<div class="hero-conf">{pct(result.predictions[0].confidence)}</div>
			</div>

			<!-- Confidence bars -->
			<div class="card">
				<h2>Top-{result.predictions.length} 예측 결과</h2>
				<div class="pred-list">
					{#each result.predictions as pred}
						<div class="pred-row" class:pred-top={pred.rank === 1}>
							<span class="pred-rank">#{pred.rank}</span>
							<div class="pred-body">
								<div class="pred-name">{pred.label}</div>
								<div class="bar-row">
									<div class="bar-track">
										<div
											class="bar-fill"
											class:bar-first={pred.rank === 1}
											style="width:{pred.confidence * 100}%"
										></div>
									</div>
									<span class="bar-pct">{pct(pred.confidence)}</span>
								</div>
							</div>
						</div>
					{/each}
				</div>

				<div class="model-footer">
					<span>Model: <strong>{result.model_info.model}</strong></span>
					{#if result.model_info.attn !== 'none'}
						<span>Attention: <strong>{result.model_info.attn.toUpperCase()}</strong></span>
					{/if}
					<span>Label: <strong>{result.model_info.label_type}</strong></span>
					<span>Classes: <strong>{result.model_info.num_classes}</strong></span>
				</div>
			</div>

		{:else}
			<div class="state-box">
				<div style="font-size:56px">✈</div>
				<p style="font-weight:600;font-size:16px;color:#475569">이미지와 모델을 선택하세요</p>
				<p style="font-size:13px;color:#94a3b8">추론 실행 버튼을 클릭하면 결과가 여기에 표시됩니다</p>
			</div>
		{/if}
	</div>
</div>

<style>
	.demo-layout {
		display: grid;
		grid-template-columns: 360px 1fr;
		gap: 20px;
		align-items: start;
	}
	@media (max-width: 780px) {
		.demo-layout { grid-template-columns: 1fr; }
	}

	.col-controls, .col-results {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}

	/* Image preview */
	.preview-img {
		width: 100%;
		max-height: 260px;
		object-fit: contain;
		background: #f1f5f9;
		border-radius: 6px;
		display: block;
	}

	/* Model metadata tags */
	.meta-tags {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		margin-top: 10px;
	}
	.tag {
		padding: 2px 10px;
		border-radius: 999px;
		font-size: 11px;
		font-weight: 600;
	}
	.tag-model { background: #dbeafe; color: #1d4ed8; }
	.tag-attn  { background: #fef9c3; color: #854d0e; }
	.tag-label { background: #dcfce7; color: #15803d; }
	.tag-data  { background: #f3e8ff; color: #7e22ce; }
	.tag-fold  { background: #e2e8f0; color: #475569; }

	/* Run button */
	.btn-run {
		width: 100%;
		padding: 13px;
		font-size: 15px;
		font-weight: 600;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
	}
	.btn-spinner {
		width: 14px;
		height: 14px;
		border: 2px solid rgba(255,255,255,0.4);
		border-top-color: white;
		border-radius: 50%;
		animation: spin 0.7s linear infinite;
		display: inline-block;
	}

	/* State boxes */
	.state-box {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		min-height: 300px;
		gap: 8px;
		text-align: center;
		color: #64748b;
	}
	.spinner-lg {
		width: 40px;
		height: 40px;
		border: 3px solid #e2e8f0;
		border-top-color: #3b82f6;
		border-radius: 50%;
		animation: spin 0.8s linear infinite;
	}
	@keyframes spin { to { transform: rotate(360deg); } }

	/* Hero card */
	.hero-card {
		background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
		color: white;
		text-align: center;
	}
	.hero-eyebrow {
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.1em;
		opacity: 0.75;
	}
	.hero-label {
		font-size: 26px;
		font-weight: 700;
		margin: 10px 0 6px;
		line-height: 1.2;
	}
	.hero-conf {
		font-size: 20px;
		font-weight: 600;
		opacity: 0.9;
	}

	/* Prediction list */
	.pred-list {
		display: flex;
		flex-direction: column;
		gap: 14px;
	}
	.pred-row {
		display: flex;
		align-items: flex-start;
		gap: 10px;
	}
	.pred-rank {
		font-size: 12px;
		font-weight: 700;
		color: #94a3b8;
		width: 26px;
		flex-shrink: 0;
		padding-top: 3px;
	}
	.pred-top .pred-rank { color: #3b82f6; }
	.pred-body { flex: 1; min-width: 0; }
	.pred-name {
		font-size: 13px;
		margin-bottom: 5px;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.pred-top .pred-name { font-weight: 600; color: #1d4ed8; }

	.bar-row {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.bar-track {
		flex: 1;
		height: 7px;
		background: #f1f5f9;
		border-radius: 999px;
		overflow: hidden;
	}
	.bar-fill {
		height: 100%;
		background: #cbd5e1;
		border-radius: 999px;
		transition: width 0.5s ease;
	}
	.bar-fill.bar-first { background: #3b82f6; }
	.bar-pct {
		font-size: 12px;
		color: #64748b;
		width: 46px;
		text-align: right;
		flex-shrink: 0;
	}

	/* Model info footer */
	.model-footer {
		display: flex;
		flex-wrap: wrap;
		gap: 14px;
		margin-top: 16px;
		padding-top: 12px;
		border-top: 1px solid #f1f5f9;
		font-size: 12px;
		color: #64748b;
	}

	/* Error */
	.error-pre {
		font-size: 12px;
		color: #dc2626;
		background: #fef2f2;
		padding: 12px;
		border-radius: 6px;
		white-space: pre-wrap;
		word-break: break-all;
		margin: 0;
	}
</style>
