"""사진 한 장(또는 여러 장)을 측정해 결과 JSON 과 확인용 overlay 를 만드는 예제.

usage:
  python run_example.py 사진.png [사진2.png ...] --output 결과폴더 [--device cpu]

결과 폴더에 <image_id>.json (공통 출력 형식 + judgment)과 <image_id>_overlay.png 가 생긴다.
overlay: 초록=line, 파랑=gap, 빨강=pn line, 주황=pt line. 측정선 양 끝 = 빨간 띠 안쪽 끝.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from judgment.judge import judge
from metrology.unet_metrology import MetrologyEngine


def draw_overlay(result: dict, image_path: Path, out_path: Path) -> None:
    """측정 위치를 원본 사진 좌표로 바꿔 그린다 (대시보드에서 그릴 때와 같은 변환)."""
    scale = result["pixel_scale"]
    fx, fy = scale["factor_x"], scale["factor_y"]

    def to_x(x):
        return x * fx + scale["offset_x"] + (fx - 1) / 2

    def to_y(y):
        return y * fy + scale["offset_y"] + (fy - 1) / 2

    labels = {line["line_id"]: line["label"] for line in result.get("judgment", {}).get("lines", [])}
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for m in result["measurements"]:
        color = (0, 200, 0) if m["measurement_type"] == "line" else (0, 90, 255)
        if m["measurement_type"] == "line" and labels.get(m["line_id"]) in ("pn", "pt"):
            color = (230, 0, 0) if labels[m["line_id"]] == "pn" else (255, 140, 0)
        y = to_y(m["y_px"]) + (-4 * fy if m["measurement_type"] == "line" else 4 * fy)
        x0, x1 = to_x(m["left_x_px"]), to_x(m["right_x_px"])
        draw.line([x0, y, x1, y], fill=color, width=3)
        draw.line([x0, y - 8, x0, y + 8], fill=color, width=2)
        draw.line([x1, y - 8, x1, y + 8], fill=color, width=2)
        if m["y_fraction"] == min(r["y_fraction"] for r in result["measurements"]):
            draw.text(((x0 + x1) / 2 - 10, y - 24), m["line_id"], fill=(0, 0, 0))
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    engine = MetrologyEngine(device=args.device)  # 모델은 한 번만 로드
    for path in args.images:
        result = judge(engine.analyze({"image_id": path.stem, "image_path": str(path)}))
        (args.output / f"{path.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["status"] == "success":
            draw_overlay(result, path, args.output / f"{path.stem}_overlay.png")
            for kind in ("line", "gap"):
                values = [m["width_um"] for m in result["measurements"] if m["measurement_type"] == kind]
                print(f"{path.name}: {kind} {len(values)}개, 평균 {sum(values) / len(values):.3f} µm")
            for line in result["judgment"]["lines"]:
                if line["label"] != "normal" or line["near_threshold"] or line["pitch_ok"] is False:
                    print(f"    {line['line_id']}: {line['label']} ({line['reason'] or 'near threshold'})"
                          f"{'' if line['pitch_ok'] is not False else ', pitch 이상'}")
        else:
            print(f"{path.name}: 실패 {result['error_code']} - {result['error_message']}")
    print("저장:", args.output)


if __name__ == "__main__":
    main()
