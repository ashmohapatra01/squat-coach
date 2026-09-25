"""Data contracts shared across the pipeline.

Keeping these as Pydantic models means the LLM narrator (app/narrator.py)
gets its output *validated* against this shape rather than trusted at face
value — it can add prose, it cannot invent a different verdict or measurement.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Result = Literal["meets", "does_not_meet", "cannot_assess"]
Assessability = Literal["conditional", "proxy_only", "cannot"]


class Measurement(BaseModel):
    """Raw numeric evidence behind a finding. Fields are optional because a
    cannot_assess finding legitimately has none of them."""

    value: Optional[float] = None
    unit: Optional[str] = None
    reference_value: Optional[float] = None
    uncertainty: Optional[float] = None
    extra: dict = Field(default_factory=dict)


class Finding(BaseModel):
    rep: int
    criterion_id: str
    criterion_title: str
    result: Result
    timestamp_seconds: Optional[float] = None
    measurement: Measurement = Field(default_factory=Measurement)
    confidence: Optional[float] = None
    source_section: str
    source_pages: list = Field(default_factory=list)
    reason: str
    feedback: str


class Rep(BaseModel):
    index: int
    start_frame: int
    bottom_frame: int
    end_frame: int
    start_time: float
    bottom_time: float
    end_time: float


class SessionResult(BaseModel):
    video_name: str
    duration_seconds: float
    fps: float
    reps_detected: int
    findings: list[Finding] = Field(default_factory=list)
    overall_summary: Optional[str] = None
    skill_name: str
    skill_version: str
