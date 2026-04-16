"""
이미지 ID를 입력하면 현재 적용된 bbox를 씌워서 보여주는 디버깅 도구.

사용법:
    python preview_bbox.py 1112805
    python preview_bbox.py 9000001
    python preview_bbox.py          # ID 없이 실행하면 프롬프트로 입력
"""

import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_IMAGES = os.path.join(BASE_DIR, "data", "fgvc-aircraft-2013b", "data", "images")
IMAGES_BOX_TXT  = os.path.join(BASE_DIR, "data", "fgvc-aircraft-2013b", "data", "images_box.txt")
USER_IMAGES     = os.path.join(BASE_DIR, "labeling-tool", "data", "user_images")
ADDED_IMAGES    = os.path.join(BASE_DIR, "labeling-tool", "data", "added_images")


def load_bbox_map():
    """images_box.txt → {id: (xmin, ymin, xmax, ymax)}"""
    bbox_map = {}
    with open(IMAGES_BOX_TXT, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                bbox_map[parts[0]] = tuple(int(x) for x in parts[1:5])
    return bbox_map


def find_image_and_bbox(image_id: str):
    """
    이미지 경로와 bbox를 반환.
    Returns: (image_path, bbox_tuple, source)
      - source: 'original' | 'user'
    """
    user_json = os.path.join(ADDED_IMAGES, f"{image_id}.json")
    user_img  = os.path.join(USER_IMAGES, f"{image_id}.jpg")
    orig_img  = os.path.join(ORIGINAL_IMAGES, f"{image_id}.jpg")

    # 사용자 추가 이미지
    if os.path.exists(user_json):
        with open(user_json) as f:
            data = json.load(f)
        bbox = data.get("bbox")
        if bbox:
            b = (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
        else:
            b = None
        return user_img, b, "user"

    # 원본 데이터셋 이미지
    if os.path.exists(orig_img):
        bbox_map = load_bbox_map()
        b = bbox_map.get(image_id)
        return orig_img, b, "original"

    return None, None, None


def draw_bbox(image_path: str, bbox, image_id: str, source: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    draw = ImageDraw.Draw(img)

    if bbox:
        xmin, ymin, xmax, ymax = bbox

        # 메인 박스 (빨간색, 두께 3)
        lw = max(2, min(w, h) // 200)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=lw)

        # 반투명 오버레이 효과 (별도 레이어로 내부 채우기)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        ov_draw.rectangle([xmin, ymin, xmax, ymax], fill=(255, 0, 0, 30))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0), width=lw)

        # 코너 좌표 텍스트
        label = f"({xmin}, {ymin}) → ({xmax}, {ymax})  [{xmax-xmin}×{ymax-ymin}]"
        font_size = max(12, min(w, h) // 40)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        tx, ty = xmin, max(0, ymin - font_size - 4)
        # 텍스트 배경
        bbox_text = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox_text, fill=(0, 0, 0, 180))
        draw.text((tx, ty), label, fill=(255, 255, 0), font=font)
    else:
        # bbox 없음 표시
        draw.text((10, 10), "bbox 없음", fill=(255, 0, 0))

    # 상단 정보 바
    info = f"ID: {image_id}  |  size: {w}×{h}  |  reference: {source}"
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        crop_w, crop_h = xmax - xmin, ymax - ymin
        ratio = f"  |  bbox 비율: {crop_w/w*100:.1f}%×{crop_h/h*100:.1f}%"
        info += ratio

    font_size_info = max(14, min(w, h) // 35)
    try:
        font_info = ImageFont.truetype("arial.ttf", font_size_info)
    except Exception:
        font_info = ImageFont.load_default()

    bar_h = font_size_info + 10
    info_bar = Image.new("RGB", (w, bar_h), (30, 30, 30))
    bar_draw = ImageDraw.Draw(info_bar)
    bar_draw.text((5, 5), info, fill=(255, 255, 255), font=font_info)

    result = Image.new("RGB", (w, h + bar_h))
    result.paste(info_bar, (0, 0))
    result.paste(img, (0, bar_h))

    return result


def main():
    image_id = sys.argv[1].strip() if len(sys.argv) > 1 else input("이미지 ID 입력: ").strip()

    image_path, bbox, source = find_image_and_bbox(image_id)

    if image_path is None:
        print(f"이미지를 찾을 수 없음: {image_id}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"이미지 파일 없음: {image_path}")
        sys.exit(1)

    print(f"이미지: {image_path}")
    print(f"bbox  : {bbox}")
    print(f"출처  : {source}")

    result = draw_bbox(image_path, bbox, image_id, source)
    result.show()

    # 저장 여부 선택
    save = input("\n저장하시겠습니까? [y/N] ").strip().lower()
    if save == 'y':
        out = f"preview_{image_id}.jpg"
        result.save(out, "JPEG", quality=95)
        print(f"저장됨: {out}")


if __name__ == "__main__":
    main()
