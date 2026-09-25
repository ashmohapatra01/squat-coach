"""Barbell tracking.

Per the design decision: a generic pose/object model does not reliably find
a barbell plate, so this asks the user to click the plate once on the first
frame, then tracks it with OpenCV's CSRT tracker. A lightweight periodic
check (does the tracked patch still look like a round plate edge?) flags
frames where the tracker has likely drifted, rather than silently trusting
it for the whole clip.

target_shape="bar" skips that circularity check entirely — a bare barbell
viewed from the side is a line, not a circle, so the plate-shaped check
would misread accurate tracking as low confidence. CSRT itself doesn't
need a circle; only the confidence signal was ever shape-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np

TargetShape = Literal["plate", "bar"]

INIT_BOX_HALF_SIZE_PLATE = 28  # px, around the user's click, at target_height resolution
INIT_BOX_HALF_SIZE_BAR = 15    # px — a thin bar needs a tighter box to avoid grabbing background
REVERIFY_EVERY_N_FRAMES = 10
MIN_CIRCULARITY_SCORE = 0.15  # heuristic threshold; only applies when target_shape="plate"


@dataclass
class BarPoint:
    frame_index: int
    x: Optional[float]
    y: Optional[float]
    confidence: float  # 0-1; low means "don't trust this frame's bar position"
    flagged: bool


def _make_tracker():
    # OpenCV moved tracker constructors around across versions/builds.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError(
        "OpenCV build has no CSRT tracker available. Install opencv-contrib-python."
    )


def _circularity_score(gray_patch: np.ndarray) -> float:
    """Cheap plausibility check, not a real detector: does this patch have a
    strong circular edge, consistent with a plate rim? Returns 0-1.
    Only meaningful for target_shape="plate" — do not call for "bar"."""
    edges = cv2.Canny(gray_patch, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    perimeter = cv2.arcLength(best, True)
    if perimeter == 0:
        return 0.0
    # circularity = 4*pi*area / perimeter^2 ; 1.0 == perfect circle
    return float(min(4 * np.pi * area / (perimeter**2), 1.0))


def track_bar(
    frames: list[np.ndarray],
    init_point: tuple[float, float],
    target_shape: TargetShape = "plate",
) -> list[BarPoint]:
    """frames: list of BGR frames in display order.
    init_point: (x, y) the user clicked on frames[0].
    target_shape: "plate" (default, uses the circularity re-verification
    check) or "bar" (bare barbell, no plate — skips the shape check and
    trusts CSRT's own success/failure signal for confidence instead)."""
    if not frames:
        return []

    half_size = INIT_BOX_HALF_SIZE_BAR if target_shape == "bar" else INIT_BOX_HALF_SIZE_PLATE

    tracker = _make_tracker()
    x0, y0 = init_point
    box = (
        max(int(x0 - half_size), 0),
        max(int(y0 - half_size), 0),
        half_size * 2,
        half_size * 2,
    )
    tracker.init(frames[0], box)

    results: list[BarPoint] = [BarPoint(0, x0, y0, confidence=1.0, flagged=False)]

    for i in range(1, len(frames)):
        ok, bbox = tracker.update(frames[i])
        if not ok:
            results.append(BarPoint(i, None, None, confidence=0.0, flagged=True))
            continue

        bx, by, bw, bh = bbox
        cx, cy = bx + bw / 2, by + bh / 2

        if target_shape == "bar":
            # No shape check for a bare bar — trust CSRT's own success signal.
            # A successful update is treated as reasonably confident; there's
            # no independent verification step for this mode yet.
            confidence = 0.75
            flagged = False
        else:
            confidence = 0.75  # base confidence for a successful CSRT update
            if i % REVERIFY_EVERY_N_FRAMES == 0:
                x, y, w, h = (int(v) for v in bbox)
                x, y = max(x, 0), max(y, 0)
                patch = frames[i][y : y + h, x : x + w]
                if patch.size > 0:
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    score = _circularity_score(gray)
                    confidence = min(confidence, 0.4 + score)
            flagged = confidence < MIN_CIRCULARITY_SCORE + 0.4

        results.append(BarPoint(i, float(cx), float(cy), confidence=confidence, flagged=flagged))

    return results
