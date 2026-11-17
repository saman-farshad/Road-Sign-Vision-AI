"""Road Sign Vision AI - a small Python/OpenCV traffic sign project.

Examples:
    python app.py
    python app.py --source assets/demo_road.png
    python app.py --source data/road_video.mp4
"""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from detector import SignClassifier, find_candidates
from sign_info import get_sign_info

BASE_DIR = Path(__file__).resolve().parent


def write_wrapped_text(frame: np.ndarray, message: str, x: int, y: int,
                       max_width: int, color: tuple[int, int, int],
                       size: float = 0.56, max_lines: int = 3) -> None:
    """Write a short English message without overflowing the side panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = message.split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        length = cv2.getTextSize(candidate, font, size, 1)[0][0]
        if line and length > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for offset, text in enumerate(lines[:max_lines]):
        cv2.putText(frame, text, (x, y + offset * 24), font, size,
                    color, 1, cv2.LINE_AA)


def make_panel(frame: np.ndarray, detection: dict | None, width: int = 370) -> np.ndarray:
    """Place a live camera on the left and the actual sign crop on the right."""
    h, w = frame.shape[:2]
    panel = np.full((h, width, 3), (28, 24, 21), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, 58), (46, 34, 29), -1)
    cv2.putText(panel, "ROAD SIGN VISION AI", (18, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (245, 229, 218), 2, cv2.LINE_AA)
    cv2.putText(panel, "RECOGNIZED SIGN", (18, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.51, (175, 178, 185), 1, cv2.LINE_AA)

    # Fixed-size sign preview box, with proper aspect ratio.
    preview_top = 106
    preview_bottom = min(h - 155, 330)
    if preview_bottom < preview_top + 55:
        preview_bottom = min(h - 100, preview_top + 80)
    cv2.rectangle(panel, (22, preview_top), (width - 22, preview_bottom),
                  (43, 39, 35), -1)

    if detection is not None:
        crop = detection["crop"]
        box_h = preview_bottom - preview_top - 16
        box_w = width - 60
        scale = min(box_w / crop.shape[1], box_h / crop.shape[0])
        sw = max(1, int(crop.shape[1] * scale))
        sh = max(1, int(crop.shape[0] * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        small = cv2.resize(crop, (sw, sh), interpolation=interpolation)
        yy = preview_top + (preview_bottom-preview_top-sh)//2
        xx = (width-sw)//2
        panel[yy:yy+sh, xx:xx+sw] = small
        label = detection["name"]
        info = detection["warning"]
        confidence = detection["confidence"]
    else:
        cv2.putText(panel, "NO SIGN DETECTED", (48, (preview_top + preview_bottom)//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (164, 170, 178), 1, cv2.LINE_AA)
        label = "No sign detected"
        info = "Waiting for a clear road sign"
        confidence = 0

    # All panel messages are English. The warning icon is drawn with OpenCV.
    info_y = min(preview_bottom + 38, h - 115)
    write_wrapped_text(panel, label, 22, info_y, width - 42,
                       (245, 245, 245), size=0.66, max_lines=2)

    icon_x, icon_y = 37, min(info_y + 58, h - 70)
    triangle = np.array([[icon_x, icon_y - 13], [icon_x - 13, icon_y + 10],
                         [icon_x + 13, icon_y + 10]], dtype=np.int32)
    cv2.polylines(panel, [triangle], True, (25, 196, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "!", (icon_x - 3, icon_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 196, 255), 2, cv2.LINE_AA)
    write_wrapped_text(panel, info, 58, icon_y + 4, width - 75,
                       (100, 211, 255), size=0.53, max_lines=2)

    cv2.putText(panel, f"CONFIDENCE  {confidence * 100:.0f}%", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.51, (170, 190, 202), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (120, 150, 180), 2)
    return np.hstack((frame, panel))


def annotate(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    result = frame.copy()
    for item in detections:
        box = item["box"]
        cv2.rectangle(result, (box.x1, box.y1), (box.x2, box.y2),
                      (40, 220, 80), 2)
        label = f'{item["name"]}  {item["confidence"]:.0%}'
        cv2.putText(result, label[:38], (box.x1, max(19, box.y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.51, (30, 245, 95), 2, cv2.LINE_AA)
    # Keep the overlay below any title printed on the synthetic demo image.
    h = result.shape[0]
    cv2.rectangle(result, (10, h - 46), (385, h - 10), (31, 30, 28), -1)
    cv2.putText(result, "Q = Quit  |  S = Screenshot", (18, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def process_frame(frame: np.ndarray, classifier: SignClassifier,
                  min_confidence: float) -> list[dict]:
    boxes = find_candidates(frame)
    crops = [frame[b.y1:b.y2, b.x1:b.x2].copy() for b in boxes]
    predictions = classifier.classify(crops)
    results = []
    for box, crop, (class_id, raw_name, confidence) in zip(boxes, crops, predictions):
        if confidence < min_confidence:
            continue
        name, warning = get_sign_info(class_id, raw_name)
        results.append({"box": box, "crop": crop, "class_id": class_id,
                        "raw_name": raw_name, "confidence": confidence,
                        "name": name, "warning": warning})
    return sorted(results, key=lambda item: item["confidence"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect German road signs in video, image or webcam")
    parser.add_argument("--source", default="0", help="Camera number (0), video or image path")
    parser.add_argument("--confidence", type=float, default=0.80,
                        help="Minimum classification confidence, between 0 and 1")
    parser.add_argument("--scan-every", type=int, default=2, help="Classify every Nth frame")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or args.scan_every < 1:
        parser.error("--confidence must be 0..1 and --scan-every >= 1")

    source = args.source
    image_path = Path(source)
    is_image = image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if is_image:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {image_path}")
        cap = None
    else:
        source_value = int(source) if source.isdecimal() else source
        cap = cv2.VideoCapture(source_value)
        if not cap.isOpened():
            raise SystemExit("Cannot open camera/video. Try --source 0 or check the video path.")

    try:
        classifier = SignClassifier()
    except Exception as error:
        if cap is not None:
            cap.release()
        raise SystemExit(f"Could not load model. Check internet and dependencies.\n{error}")

    save_dir = BASE_DIR / "outputs"
    save_dir.mkdir(exist_ok=True)
    last = []
    count = 0
    try:
        while True:
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break

            if frame.shape[1] > 1280:
                ratio = 1280 / frame.shape[1]
                frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
            # Keep layout readable even if source video is low-resolution.
            if frame.shape[0] < 390:
                ratio = 390 / frame.shape[0]
                frame = cv2.resize(frame, None, fx=ratio, fy=ratio)
            count += 1
            if is_image or count % args.scan_every == 1 or args.scan_every == 1:
                last = process_frame(frame, classifier, args.confidence)
            current = last[0] if last else None
            output = make_panel(annotate(frame, last), current)
            cv2.imshow("Road Sign Vision AI", output)
            key = cv2.waitKey(0 if is_image else 1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("s"):
                filename = save_dir / f"road_sign_{datetime.now():%Y%m%d_%H%M%S}.png"
                cv2.imwrite(str(filename), output)
                print(f"Screenshot saved: {filename}")
            if is_image:
                # Still image is displayed once until a key is pressed.
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
