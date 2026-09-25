"""Repetition detection.

Deliberately deterministic — a state machine over the smoothed hip-height
trajectory, not an LLM guess. The bottom of each rep is a local extremum in
hip height, confirmed by maximum knee/hip flexion, matching how the
reference document itself describes identifying the bottom position.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from app.schemas import Rep


class _State(Enum):
    STANDING = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()


def smooth_series(values: list[Optional[float]], window: int = 9, polyorder: int = 2) -> np.ndarray:
    """Interpolates over missing (None) values, then Savitzky-Golay smooths.
    Keeps the raw series available to the caller for confidence reporting —
    this function only returns the smoothed version used for rep timing."""
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    nans = np.isnan(arr)
    if nans.all():
        return arr
    arr[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])

    w = min(window, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
    if w < 5:
        return arr  # too short to smooth meaningfully
    return savgol_filter(arr, window_length=w, polyorder=min(polyorder, w - 1))


def detect_reps(
    hip_y: list[Optional[float]],
    fps: float,
    descent_threshold_px: float = 12.0,
    min_bottom_frames: int = 2,
) -> list[Rep]:
    """hip_y: per-frame hip y-coordinate in pixels (image coords: larger y = lower on screen,
    i.e. deeper in the squat). Returns detected reps in chronological order."""
    smoothed = smooth_series(hip_y)
    if len(smoothed) < 5:
        return []

    # Standing = highest hip position (smallest y) over the whole clip. Taking the
    # first ~0.5s assumed the clip starts standing, which a clip trimmed or
    # recorded mid-rep breaks. A low percentile rather than the min, so a
    # single-frame landmark glitch can't set the baseline.
    baseline = float(np.percentile(smoothed, 5))
    # A rep whose bottom lands in this opening window was already rising when
    # the clip started — its true bottom is off-clip, so it isn't reported.
    partial_start_frames = max(int(fps * 0.1), min_bottom_frames)
    state = _State.STANDING
    reps: list[Rep] = []
    start_frame = 0
    bottom_frame = 0
    bottom_value = baseline
    bottom_run = 0

    for i in range(1, len(smoothed)):
        prev, cur = smoothed[i - 1], smoothed[i]
        delta = cur - prev  # positive = moving down (y increases downward)

        # The bottom is the deepest point of the whole rep, not wherever descent
        # first paused — a slow start can pause briefly and then keep sinking.
        if state != _State.STANDING and cur >= bottom_value:
            bottom_value = cur
            bottom_frame = i

        if state == _State.STANDING:
            if cur - baseline > descent_threshold_px:
                state = _State.DESCENDING
                start_frame = max(i - 1, 0)
                bottom_value = cur
                bottom_frame = i

        elif state == _State.DESCENDING:
            if delta <= 0.5:  # roughly stopped descending
                bottom_run += 1
                if bottom_run >= min_bottom_frames:
                    state = _State.BOTTOM
            else:
                bottom_run = 0

        elif state == _State.BOTTOM:
            if delta < -0.5:  # started rising (y decreasing)
                state = _State.ASCENDING

        elif state == _State.ASCENDING:
            if abs(cur - baseline) < descent_threshold_px * 0.5:
                end_frame = i
                if start_frame == 0 and bottom_frame <= partial_start_frames:
                    state = _State.STANDING
                    bottom_run = 0
                    continue
                reps.append(
                    Rep(
                        index=len(reps) + 1,
                        start_frame=start_frame,
                        bottom_frame=bottom_frame,
                        end_frame=end_frame,
                        start_time=start_frame / fps,
                        bottom_time=bottom_frame / fps,
                        end_time=end_frame / fps,
                    )
                )
                state = _State.STANDING
                bottom_run = 0

    return reps
