"""Video validation and preprocessing.

Deliberately conservative: we'd rather tell the user their video won't
work up front than silently produce a garbage assessment from it.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}
MIN_DURATION_S = 5
MAX_DURATION_S = 30
TARGET_HEIGHT = 720


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


@dataclass
class ValidationResult:
    ok: bool
    reason: str
    info: Optional[VideoInfo] = None
    warning: Optional[str] = None  # non-blocking concern the UI should surface


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def validate_video(path: Path) -> ValidationResult:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return ValidationResult(False, f"Unsupported format '{path.suffix}'. Use MP4 or MOV.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return ValidationResult(False, "Could not open the video file — it may be corrupted.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0 or frame_count <= 0:
        return ValidationResult(False, "Could not read frame rate or frame count from this file.")

    duration_s = frame_count / fps

    if duration_s < MIN_DURATION_S:
        return ValidationResult(False, f"Video is only {duration_s:.1f}s — needs to be at least {MIN_DURATION_S}s.")
    if duration_s > MAX_DURATION_S:
        return ValidationResult(False, f"Video is {duration_s:.1f}s — trim to under {MAX_DURATION_S}s for this MVP.")
    if width == 0 or height == 0:
        return ValidationResult(False, "Could not determine video dimensions.")

    # Orientation sanity check: phone clips are often portrait and can still
    # frame a side-view squat fully, so warn rather than block — the tight
    # framing may cut off the bar or feet at the edges.
    warning = None
    if height > width * 1.3:
        warning = "This is a portrait video; framing may cut off the bar or feet at the edges."

    info = VideoInfo(path=path, fps=fps, frame_count=frame_count, duration_s=duration_s, width=width, height=height)
    return ValidationResult(True, "OK", info, warning)


def get_first_frame(info: VideoInfo, target_height: int = TARGET_HEIGHT):
    """Reads just the first frame without leaving a VideoCapture open —
    used for the bar-click UI step, separate from the full extract_frames pass."""
    cap = cv2.VideoCapture(str(info.path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    scale = target_height / info.height if info.height > target_height else 1.0
    if scale != 1.0:
        frame = cv2.resize(frame, (int(info.width * scale), int(info.height * scale)))
    return frame


def extract_frames(info: VideoInfo, target_height: int = TARGET_HEIGHT):
    """Yields (frame_index, timestamp_seconds, bgr_frame) resized to target_height,
    preserving aspect ratio, so the frame-to-timestamp mapping stays exact."""
    cap = cv2.VideoCapture(str(info.path))
    scale = target_height / info.height if info.height > target_height else 1.0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (int(info.width * scale), int(info.height * scale)))
        yield idx, idx / info.fps, frame
        idx += 1
    cap.release()


def reencode_for_browser(src_avi: Path, dst_mp4: Path) -> bool:
    """Re-encodes the OpenCV-written .avi into H.264 mp4 so it plays inline
    in Streamlit/browsers, via ffmpeg or else OpenCV's H.264 writer. Returns
    False (leaving the .avi in place) if neither works — caller should tell
    the user why playback failed."""
    if has_ffmpeg():
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_avi), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(dst_mp4)],
            capture_output=True,
        )
        if result.returncode == 0 and dst_mp4.exists():
            return True
    return _reencode_with_opencv(src_avi, dst_mp4)


def _reencode_with_opencv(src_avi: Path, dst_mp4: Path) -> bool:
    """Fallback when ffmpeg is missing: some OpenCV builds (e.g. macOS via
    AVFoundation) can write H.264 ('avc1') mp4 directly."""
    cap = cv2.VideoCapture(str(src_avi))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(dst_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        cap.release()
        return False
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    return written > 0 and dst_mp4.exists() and dst_mp4.stat().st_size > 0
