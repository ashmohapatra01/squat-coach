"""Pose landmark extraction via MediaPipe.

Key decision (per the reference document's own side-view assumption): we
pick whichever side of the body is more visible across the clip and use
that side consistently, rather than averaging left/right landmarks —
averaging a near-side and far-side hip produces a physically meaningless
point.
"""
from __future__ import annotations

import os
import ssl
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    _HAS_MEDIAPIPE = True
except ImportError:  # pragma: no cover
    _HAS_MEDIAPIPE = False

# mediapipe >= 0.10.30 dropped the legacy `mp.solutions` API; the Tasks API
# needs a model file. It's not shipped (gitignored, excluded from the zip) —
# ensure_model() downloads it on first use, the same self-healing pattern as
# credentials.yaml in app/auth.py.
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
MIN_MODEL_BYTES = 1_000_000  # the real model is ~9 MB; anything tiny is an error page, not a model


def ensure_model() -> Path:
    """Returns MODEL_PATH, downloading the model first if it's missing.

    Downloads to a temp file in the same folder and renames it into place only
    once complete, so an interrupted download (or two sessions downloading at
    once) can never leave a partial file that later passes the exists() check."""
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  # git doesn't keep the empty models/ folder
    fd, tmp_name = tempfile.mkstemp(dir=MODEL_PATH.parent, suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        # python.org macOS builds often lack system CA certs; certifi ships with mediapipe.
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(MODEL_URL, timeout=60, context=context) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        if tmp_path.stat().st_size < MIN_MODEL_BYTES:
            raise RuntimeError(f"downloaded file is only {tmp_path.stat().st_size} bytes")
        tmp_path.chmod(0o644)  # mkstemp creates owner-only files
        os.replace(tmp_path, MODEL_PATH)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Pose model not found at {MODEL_PATH} and the automatic download failed ({e}). "
            f"Download it manually:\n  curl -L -o {MODEL_PATH} {MODEL_URL}"
        ) from e
    return MODEL_PATH

LANDMARK_NAMES = ["shoulder", "hip", "knee", "ankle", "heel", "foot_index", "ear"]

# MediaPipe pose landmark indices, per side
_MP_INDEX = {
    "left": {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "foot_index": 31, "ear": 7},
    "right": {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "foot_index": 32, "ear": 8},
}


@dataclass
class FramePose:
    frame_index: int
    side: str  # "left" or "right"
    points: dict = field(default_factory=dict)  # name -> (x, y, visibility) in pixel coords
    mean_visibility: float = 0.0


class PoseExtractor:
    def __init__(self):
        if not _HAS_MEDIAPIPE:
            raise RuntimeError(
                "mediapipe is not installed. Run `pip install -r requirements.txt` "
                "inside your virtualenv first."
            )
        ensure_model()
        # VIDEO mode tracks across consecutive frames (needs increasing
        # timestamps); IMAGE mode is used for the non-consecutive side-choice samples.
        self._video = self._create(vision.RunningMode.VIDEO)
        self._image = self._create(vision.RunningMode.IMAGE)
        self._last_ts_ms = -1

    @staticmethod
    def _create(mode):
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def close(self):
        self._video.close()
        self._image.close()

    def _raw_landmarks(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None):
        """Returns a list of 33 landmarks (with .x/.y/.visibility) or None."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if timestamp_ms is None:
            result = self._image.detect(image)
        else:
            timestamp_ms = max(timestamp_ms, self._last_ts_ms + 1)
            self._last_ts_ms = timestamp_ms
            result = self._video.detect_for_video(image, timestamp_ms)
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def choose_side(self, sample_frames: list[np.ndarray]) -> str:
        """Runs pose on a handful of sample frames and returns whichever
        side ("left"/"right") had higher average landmark visibility."""
        totals = {"left": 0.0, "right": 0.0}
        counts = {"left": 0, "right": 0}
        for frame in sample_frames:
            lm = self._raw_landmarks(frame)
            if lm is None:
                continue
            for side, idx_map in _MP_INDEX.items():
                vis = [lm[i].visibility or 0.0 for i in idx_map.values()]
                totals[side] += float(np.mean(vis))
                counts[side] += 1
        left_avg = totals["left"] / max(counts["left"], 1)
        right_avg = totals["right"] / max(counts["right"], 1)
        return "left" if left_avg >= right_avg else "right"

    def extract(self, frame_index: int, frame_bgr: np.ndarray, side: str, timestamp_s: float = 0.0) -> Optional[FramePose]:
        h, w = frame_bgr.shape[:2]
        lm = self._raw_landmarks(frame_bgr, int(timestamp_s * 1000))
        if lm is None:
            return None

        idx_map = _MP_INDEX[side]
        points = {}
        visibilities = []
        for name, i in idx_map.items():
            p = lm[i]
            vis = p.visibility or 0.0
            points[name] = (p.x * w, p.y * h, vis)
            visibilities.append(vis)

        return FramePose(
            frame_index=frame_index,
            side=side,
            points=points,
            mean_visibility=float(np.mean(visibilities)),
        )
