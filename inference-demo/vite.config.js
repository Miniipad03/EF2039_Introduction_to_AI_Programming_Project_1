import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		watch: {
			// 서버 환경에서 inotify 한도 초과 방지 — src/ 외부는 감시 제외
			ignored: [
				'**/node_modules/**',
				'**/.svelte-kit/**',
				'**/scripts/**',
				'**/*.pth',
				'**/*.csv',
				'**/data/**',
				'**/__pycache__/**',
			],
		},
	},
});
