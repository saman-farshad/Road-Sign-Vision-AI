import cv2
import numpy as np

from detector import find_candidates


def test_red_circle_is_a_candidate():
    frame = np.full((480, 800, 3), 230, dtype=np.uint8)
    cv2.circle(frame, (630, 155), 55, (0, 0, 255), 12)
    found = find_candidates(frame)
    assert any(c.x1 < 630 < c.x2 and c.y1 < 155 < c.y2 for c in found)


def test_plain_frame_has_no_candidates():
    frame = np.full((480, 800, 3), 200, dtype=np.uint8)
    assert find_candidates(frame) == []
