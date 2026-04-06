<script>
	import '../app.css';
	import { page } from '$app/stores';
</script>

<nav>
	<span class="logo">✈ FGVC Labeler</span>
	<a href="/" class:active={$page.url.pathname === '/'}>통계</a>
	<a href="/browse" class:active={$page.url.pathname === '/browse'}>이미지 브라우저</a>
	<a href="/excluded" class:active={$page.url.pathname === '/excluded'}>제외된 이미지</a>
	<a href="/upload" class:active={$page.url.pathname === '/upload'}>이미지 추가</a>
</nav>

<main>
	<slot />
</main>
