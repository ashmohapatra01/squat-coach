"""In-app video trimming — cut a clip down (e.g. to fit the 5-30s MVP limit,
or to isolate a specific set) without leaving the app.

Additive: does not modify video.py. Reuses reencode_for_browser() from
there, which already has the ffmpeg-or-OpenCV-H264-fallback logic proven
working on this machine — no new external dependency needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from app.video import reencode_for_browser


@dataclass
class TrimResult:
    ok: bool
    reason: str
    path: Optional[Path] = None


def probe_duration(path: Path) -> Optional[tuple[float, float, int, int]]:
    """Returns (fps, duration_s, width, height) with none of validate_video()'s
    pass/fail rules applied — just the raw numbers, so the UI can offer
    trimming before deciding whether to accept or reject a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        return None
    return fps, frame_count / fps, width, height


def trim_video(src: Path, start_s: float, end_s: float, dest_dir: Path) -> TrimResult:
    """Cuts [start_s, end_s) out of src and writes a new file under dest_dir.

    Frame-accurate: reads and re-writes every frame in range rather than
    doing a fast keyframe-only cut. Slightly slower than ffmpeg's `-c copy`
    would be, but correct — and consistent with the rest of this app, which
    doesn't assume ffmpeg is installed."""
    if end_s <= start_s:
        return TrimResult(False, "End time must be after start time.")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return TrimResult(False, "Could not open the source video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)

    dest_dir.mkdir(parents=True, exist_ok=True)
    avi_path = dest_dir / f"{src.stem}_trimmed_raw.avi"
    # Opened on the first in-range frame and sized from it, not from the
    # container's reported width/height — if those ever disagree (e.g. rotation
    # metadata), VideoWriter silently drops every mismatched frame.
    writer = None

    idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start_frame <= idx < end_frame:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            writer.write(frame)
            written += 1
        if idx >= end_frame:
            break
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()

    if written == 0:
        avi_path.unlink(missing_ok=True)
        return TrimResult(False, "No frames fell inside the selected range.")

    mp4_path = dest_dir / f"{src.stem}_trimmed.mp4"
    ok = reencode_for_browser(avi_path, mp4_path)
    avi_path.unlink(missing_ok=True)

    if ok:
        return TrimResult(True, f"Trimmed to {written} frames (~{written / fps:.1f}s).", mp4_path)
    return TrimResult(
        False,
        "Trim succeeded but the browser-compatible re-encode step failed "
        "(no ffmpeg and no OpenCV H.264 fallback available on this machine).",
    )
