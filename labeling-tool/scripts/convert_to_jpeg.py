"""
업로드된 이미지를 JPEG로 in-place 변환하는 헬퍼 스크립트.
사용법: python convert_to_jpeg.py <file_path>
- JPEG면 아무것도 안 함 (0 exit)
- AVIF/WebP/PNG면 JPEG로 덮어씀 (0 exit)
- 실패 시 stderr에 오류 메시지 출력 후 1 exit
"""

import sys
import os
from PIL import Image


def detect_format(path):
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


def to_rgb_white_bg(img):
    if img.mode in ('RGBA', 'LA'):
        bg = Image.new('RGB', img.size, (255, 255, 255))
        mask = img.split()[-1]
        bg.paste(img.convert('RGB'), mask=mask)
        return bg
    if img.mode == 'PA':
        rgba = img.convert('RGBA')
        bg = Image.new('RGB', rgba.size, (255, 255, 255))
        bg.paste(rgba.convert('RGB'), mask=rgba.split()[3])
        return bg
    return img.convert('RGB')


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_jpeg.py <file_path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    fmt = detect_format(path)
    if fmt == 'JPEG':
        sys.exit(0)  # 이미 JPEG, 아무것도 안 함

    if fmt == 'UNKNOWN':
        print(f"Unsupported format: {path}", file=sys.stderr)
        sys.exit(1)

    img = Image.open(path)
    img.load()
    rgb = to_rgb_white_bg(img)
    rgb.save(path, 'JPEG', quality=95, subsampling=0)


if __name__ == '__main__':
    main()
