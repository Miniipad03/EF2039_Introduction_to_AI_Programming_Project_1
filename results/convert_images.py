"""
Non-JPEG 이미지(AVIF, WebP, PNG)를 JPEG로 변환하는 스크립트.
- Pillow 내장 디코더 사용 → ffmpeg의 AVIF 타일 패딩 버그 없음
- 투명 채널(RGBA/LA/PA) → 흰색 배경으로 합성
- 변환 대상 파일을 in-place 덮어씀 (파일명 유지)
"""

import os
from PIL import Image

IMG_DIR = os.path.join(os.path.dirname(__file__),
                       "labeling-tool", "data", "user_images")
JPEG_QUALITY = 95

MAGIC_SIGNATURES = {
    b'\xff\xd8': 'JPEG',
}

def detect_format(path: str) -> str:
    """파일 헤더의 magic bytes로 실제 포맷을 판별."""
    with open(path, 'rb') as f:
        header = f.read(16)
    if header[:2] == b'\xff\xd8':
        return 'JPEG'
    if b'avif' in header or b'avis' in header:
        return 'AVIF'
    if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
        return 'WEBP'
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return 'PNG'
    return 'UNKNOWN'


def to_rgb_white_bg(img: Image.Image) -> Image.Image:
    """투명 채널이 있는 이미지를 흰색 배경 RGB로 변환."""
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            bg.paste(img.convert('RGB'), mask=img.split()[3])
        else:  # LA
            bg.paste(img.convert('RGB'), mask=img.split()[1])
        return bg
    if img.mode == 'PA':
        img = img.convert('RGBA')
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img.convert('RGB'), mask=img.split()[3])
        return bg
    return img.convert('RGB')


def convert_file(path: str, fmt: str) -> None:
    """단일 파일을 JPEG로 변환하여 in-place 덮어씀."""
    img = Image.open(path)
    img.load()  # 완전히 디코딩 (lazy loading 방지)
    rgb = to_rgb_white_bg(img)
    rgb.save(path, 'JPEG', quality=JPEG_QUALITY, subsampling=0)
    print(f"  변환 완료: {os.path.basename(path)}  {img.size}  {fmt} → JPEG")


def main():
    if not os.path.isdir(IMG_DIR):
        print(f"디렉터리를 찾을 수 없음: {IMG_DIR}")
        return

    files = sorted(os.listdir(IMG_DIR))
    targets = []

    print("=== 변환 대상 파일 탐색 ===")
    for fname in files:
        if fname.startswith('.'):
            continue
        path = os.path.join(IMG_DIR, fname)
        if not os.path.isfile(path):
            continue
        fmt = detect_format(path)
        if fmt != 'JPEG':
            targets.append((path, fmt))
            print(f"  발견: {fname}  ({fmt})")

    if not targets:
        print("변환할 파일 없음 — 모두 JPEG입니다.")
        return

    print(f"\n=== 총 {len(targets)}개 파일 변환 시작 ===")
    errors = []
    for path, fmt in targets:
        try:
            convert_file(path, fmt)
        except Exception as e:
            print(f"  오류: {os.path.basename(path)} — {e}")
            errors.append((path, e))

    print(f"\n=== 완료: {len(targets) - len(errors)}개 성공"
          + (f", {len(errors)}개 실패" if errors else "") + " ===")
    for path, e in errors:
        print(f"  실패: {os.path.basename(path)}: {e}")


if __name__ == '__main__':
    main()
