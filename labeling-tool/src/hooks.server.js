// 서버 시작 시 데이터셋 annotation 파일 미리 파싱
// → 첫 브라우저 요청도 캐시 히트
import { getImageMap } from '$lib/datastore.js';
getImageMap();
