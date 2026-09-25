"""Deterministic assessment.

This is the rule engine: skill file + computed measurements in, Findings
out. No LLM involved here — the narrator (app/narrator.py) only phrases
what this module has already decided. Any numeric tolerance used below
(e.g. how many pixels of bar drift counts as "off midfoot") is an
*application heuristic*, called out as such, not a document requirement —
the reference document gives relationships, not universal thresholds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from app.measurements import (
    BackAngleMeasurement,
    BarMidfootMeasurement,
    DepthMeasurement,
    HipDriveMeasurement,
)
from app.schemas import Finding, Measurement

# --- application heuristics (NOT document requirements) -------------------
BAR_MIDFOOT_TOLERANCE_PX = 25.0
HIP_DRIVE_MIN_LEAD_FRAMES = 0  # hips must not trail the shoulders
# ---------------------------------------------------------------------------


def load_skill(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _criterion(skill: dict, criterion_id: str) -> dict:
    for c in skill["criteria"]:
        if c["id"] == criterion_id:
            return c
    raise KeyError(f"Unknown criterion id: {criterion_id}")


def _finding(
    rep_index: int,
    criterion: dict,
    result: str,
    timestamp: Optional[float],
    measurement: Measurement,
    confidence: Optional[float],
    reason: str,
) -> Finding:
    feedback_map = criterion["feedback"]
    feedback = feedback_map.get(result) or feedback_map.get("fail") or feedback_map.get("cannot_assess", "")
    return Finding(
        rep=rep_index,
        criterion_id=criterion["id"],
        criterion_title=criterion["title"],
        result=result,
        timestamp_seconds=timestamp,
        measurement=measurement,
        confidence=confidence,
        source_section=criterion["source_section"],
        source_pages=criterion.get("source_pages", []),
        reason=reason,
        feedback=feedback,
    )


def assess_depth(skill: dict, rep_index: int, timestamp: float, m: DepthMeasurement) -> Finding:
    c = _criterion(skill, "squat_depth")
    if m.confidence < 0.5 or m.delta_px is None:
        return _finding(
            rep_index, c, "cannot_assess", timestamp,
            Measurement(uncertainty=1 - m.confidence),
            m.confidence, "Hip or knee landmark confidence too low at the bottom frame.",
        )
    result = "meets" if m.delta_px > 0 else "does_not_meet"
    return _finding(
        rep_index, c, result, timestamp,
        Measurement(value=m.delta_px, unit="px", reference_value=0.0),
        m.confidence,
        f"Hip crease is {'below' if m.delta_px > 0 else 'above'} the top of the patella by {abs(m.delta_px):.0f}px.",
    )


def assess_bar_midfoot(skill: dict, rep_index: int, timestamp: float, m: BarMidfootMeasurement) -> Finding:
    c = _criterion(skill, "bar_midfoot_alignment")
    if m.confidence < 0.5 or m.offset_px is None:
        return _finding(
            rep_index, c, "cannot_assess", timestamp,
            Measurement(uncertainty=1 - m.confidence),
            m.confidence, "Bar tracking confidence or foot-landmark visibility too low.",
        )
    result = "meets" if abs(m.offset_px) <= BAR_MIDFOOT_TOLERANCE_PX else "does_not_meet"
    return _finding(
        rep_index, c, result, timestamp,
        Measurement(value=m.offset_px, unit="px", reference_value=0.0, uncertainty=BAR_MIDFOOT_TOLERANCE_PX),
        m.confidence,
        f"Bar is {abs(m.offset_px):.0f}px {'forward' if m.offset_px > 0 else 'behind'} the midfoot "
        f"(app tolerance: {BAR_MIDFOOT_TOLERANCE_PX:.0f}px, not a document threshold).",
    )


def assess_hip_drive(skill: dict, rep_index: int, timestamp: float, m: HipDriveMeasurement) -> Finding:
    c = _criterion(skill, "hip_drive")
    if m.confidence < 0.5 or m.hip_lead_frames is None:
        return _finding(
            rep_index, c, "cannot_assess", timestamp,
            Measurement(uncertainty=1 - m.confidence),
            m.confidence, "Could not reliably find a 'starts rising' frame for hip and/or shoulder trajectory.",
        )
    result = "meets" if m.hip_lead_frames >= HIP_DRIVE_MIN_LEAD_FRAMES else "does_not_meet"
    return _finding(
        rep_index, c, result, timestamp,
        Measurement(value=float(m.hip_lead_frames), unit="frames", reference_value=0.0),
        m.confidence,
        f"Hips began rising {abs(m.hip_lead_frames)} frame(s) "
        f"{'before' if m.hip_lead_frames >= 0 else 'after'} the shoulders.",
    )


def assess_back_angle(skill: dict, rep_index: int, timestamp: float, m: BackAngleMeasurement) -> Finding:
    c = _criterion(skill, "back_angle_estimate")
    if m.confidence < 0.5 or m.angle_degrees is None:
        return _finding(
            rep_index, c, "cannot_assess", timestamp,
            Measurement(uncertainty=1 - m.confidence),
            m.confidence, "Shoulder or hip landmark confidence too low.",
        )
    # report_only in the skill file: no meets/does_not_meet verdict, just the estimate.
    return _finding(
        rep_index, c, "meets", timestamp,
        Measurement(value=m.angle_degrees, unit="degrees"),
        m.confidence, f"Estimated back angle at the bottom: {m.angle_degrees:.0f}° from horizontal (reported, not scored).",
    )


def assess_not_observable(skill: dict, rep_index: int, criterion_id: str) -> Finding:
    """For criteria the skill file marks assessability: cannot — always
    cannot_assess, regardless of what the video shows, because a side-view
    clip structurally cannot answer this."""
    c = _criterion(skill, criterion_id)
    return _finding(
        rep_index, c, "cannot_assess", None,
        Measurement(), None,
        f"Requires {c.get('camera_requirement', 'a different camera angle')}, not available from a single side-view video.",
    )
