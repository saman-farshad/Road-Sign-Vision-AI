"""Find likely sign shapes with OpenCV, then classify their cropped images."""

from dataclasses import dataclass

import cv2
import numpy as np

from sign_info import GTSRB_LABELS

MODEL_NAME = "FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb"


@dataclass
class Candidate:
    x1: int
    y1: int
    x2: int
    y2: int


def find_candidates(frame: np.ndarray, max_candidates: int = 6) -> list[Candidate]:
    """Detect red, blue and yellow sign-like shapes; not an AI prediction."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 85, 65), (11, 255, 255))
    red2 = cv2.inRange(hsv, (168, 85, 65), (179, 255, 255))
    blue = cv2.inRange(hsv, (95, 90, 55), (135, 255, 255))
    yellow = cv2.inRange(hsv, (17, 105, 90), (38, 255, 255))
    mask = red1 | red2 | blue | yellow
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 130:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 22 or bh < 22:
            continue
        if bw > w * 0.45 or bh > h * 0.55:
            continue
        ratio = bw / bh
        if not 0.68 <= ratio <= 1.45:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.40:
            continue

        # Reject isolated colored pixels and near-empty boxes.
        colored_ratio = cv2.countNonZero(mask[y:y + bh, x:x + bw]) / (bw * bh)
        if colored_ratio < 0.10:
            continue

        padding = int(max(bw, bh) * 0.07)
        box = Candidate(max(0, x-padding), max(0, y-padding),
                        min(w, x+bw+padding), min(h, y+bh+padding))
        candidates.append((area, box))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in candidates[:max_candidates]]


class SignClassifier:
    def __init__(self):
        # Imported here to keep simple OpenCV tests independent of Hugging Face.
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.torch = torch
        print("Loading the GTSRB sign classifier (first run needs internet)...")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
        self.model.eval()

    def classify(self, crops: list[np.ndarray]) -> list[tuple[int, str, float]]:
        """Return class ID, original model label and confidence for each crop."""
        if not crops:
            return []
        from PIL import Image

        rgb_images = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                      for crop in crops]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1)
            scores, class_ids = probabilities.max(dim=-1)

        found = []
        for cls, score in zip(class_ids.tolist(), scores.tolist()):
            name = self.model.config.id2label.get(cls, GTSRB_LABELS[cls] if cls < 43 else str(cls))
            found.append((cls, name, float(score)))
        return found
