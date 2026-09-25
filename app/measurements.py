"""Turns raw landmark + bar positions into the numeric measurements each
skill criterion is checked against. Every function here returns both a
value and a confidence signal — the assessor decides pass/fail/cannot-assess,
this module just does arithmetic on pixels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

VISIBILITY_FLOOR = 0.5  # below this, treat a landmark as unusable


@dataclass
class DepthMeasurement:
    hip_crease_y: Optional[float]
    patella_top_y: Optional[float]
    delta_px: Optional[float]  # positive = hip below knee (good)
    confidence: float


def measure_depth(hip_point, knee_point) -> DepthMeasurement:
    """hip_point / knee_point: (x, y, visibility) or None."""
    if hip_point is None or knee_point is None:
        return DepthMeasurement(None, None, None, 0.0)
    hip_x, hip_y, hip_vis = hip_point
    knee_x, knee_y, knee_vis = knee_point
    confidence = min(hip_vis, knee_vis)
    if confidence < VISIBILITY_FLOOR:
        return DepthMeasurement(hip_y, knee_y, None, confidence)
    delta = hip_y - knee_y  # image y grows downward, so hip below knee => hip_y > knee_y
    return DepthMeasurement(hip_y, knee_y, delta, confidence)


@dataclass
class BarMidfootMeasurement:
    bar_x: Optional[float]
    midfoot_x: Optional[float]
    offset_px: Optional[float]  # signed distance, bar minus midfoot
    confidence: float


def measure_bar_midfoot(bar_point, ankle_point, foot_index_point, bar_confidence: float) -> BarMidfootMeasurement:
    if bar_point is None or ankle_point is None or foot_index_point is None:
        return BarMidfootMeasurement(None, None, None, 0.0)
    bar_x, _bar_y = bar_point
    ankle_x, _, ankle_vis = ankle_point
    foot_x, _, foot_vis = foot_index_point
    landmark_conf = min(ankle_vis, foot_vis)
    confidence = min(landmark_conf, bar_confidence)
    if confidence < VISIBILITY_FLOOR:
        return BarMidfootMeasurement(bar_x, None, None, confidence)
    midfoot_x = (ankle_x + foot_x) / 2
    return BarMidfootMeasurement(bar_x, midfoot_x, bar_x - midfoot_x, confidence)


@dataclass
class BackAngleMeasurement:
    angle_degrees: Optional[float]  # from horizontal
    confidence: float


def measure_back_angle(shoulder_point, hip_point) -> BackAngleMeasurement:
    if shoulder_point is None or hip_point is None:
        return BackAngleMeasurement(None, 0.0)
    sx, sy, s_vis = shoulder_point
    hx, hy, h_vis = hip_point
    confidence = min(s_vis, h_vis)
    if confidence < VISIBILITY_FLOOR:
        return BackAngleMeasurement(None, confidence)
    angle = math.degrees(math.atan2(abs(sy - hy), abs(sx - hx) or 1e-6))
    return BackAngleMeasurement(angle, confidence)


@dataclass
class HipDriveMeasurement:
    hip_lead_frames: Optional[int]  # positive = hips started rising before shoulders
    confidence: float


def measure_hip_drive(
    hip_y_series: list[Optional[float]],
    shoulder_y_series: list[Optional[float]],
    bottom_frame: int,
    search_window: int = 15,
) -> HipDriveMeasurement:
    """Finds the first frame after bottom_frame where each series starts
    reliably decreasing (rising, since image y grows downward), and reports
    how many frames apart those two 'starts rising' events are."""

    def first_rise_frame(series):
        for i in range(bottom_frame + 1, min(bottom_frame + 1 + search_window, len(series) - 1)):
            a, b = series[i], series[i + 1]
            if a is None or b is None:
                continue
            if b < a - 0.5:  # decreasing y = rising up
                return i
        return None

    hip_rise = first_rise_frame(hip_y_series)
    shoulder_rise = first_rise_frame(shoulder_y_series)

    if hip_rise is None or shoulder_rise is None:
        return HipDriveMeasurement(None, 0.3)

    return HipDriveMeasurement(hip_lead_frames=shoulder_rise - hip_rise, confidence=0.7)
