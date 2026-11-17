"""Create a clearly synthetic demo road image; it is not a model prediction."""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "assets" / "demo_road.png"


def main() -> None:
    h, w = 540, 960
    image = np.full((h, w, 3), (236, 214, 186), dtype=np.uint8)
    # Sky / distant trees / street
    cv2.rectangle(image, (0, 285), (w, 540), (105, 166, 111), -1)
    cv2.rectangle(image, (0, 0), (w, 280), (233, 199, 155), -1)
    road = np.array([[355, 285], [604, 285], [900, h], [30, h]], dtype=np.int32)
    cv2.fillPoly(image, [road], (102, 108, 111))
    for y, length in [(316, 17), (383, 42), (460, 69)]:
        cv2.line(image, (480, y), (480, y+length), (245, 245, 245), 5)

    # A standard circular German-style 30 km/h speed limit sign.
    cx, cy, r = 770, 186, 77
    cv2.line(image, (cx, cy+r), (cx, 450), (140, 140, 140), 8)
    cv2.circle(image, (cx, cy), r+2, (15, 15, 15), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), r, (250, 250, 250), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), r-8, (25, 25, 225), 13, cv2.LINE_AA)
    cv2.putText(image, "30", (cx-52, cy+24), cv2.FONT_HERSHEY_SIMPLEX, 1.75,
                (10, 10, 10), 5, cv2.LINE_AA)
    cv2.putText(image, "SYNTHETIC TEST IMAGE", (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT), image)
    print(f"Created: {OUT}")


if __name__ == "__main__":
    main()
