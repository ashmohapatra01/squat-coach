"""Orchestrates the full pipeline: pixels -> landmarks -> measurements ->
rule decision -> (optional) LLM narration -> annotated video. Deterministic
steps run first and fully decide every finding; narration only phrases
results that already exist.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app import assessor, measurements as meas
from app.bar_tracker import BarTracker, TargetShape
from app.pose import PoseExtractor
from app.rep_detector import detect_reps
from app.schemas import Finding, SessionResult
from app.video import VideoInfo, count_frames, extract_frames, reencode_for_browser

ProgressFn = Callable[[str, float], None]

DEPTH_GREEN = (95, 122, 63)  # BGR
DEPTH_RED = (58, 58, 178)
BAR_COLOR = (37, 111, 42)
SKELETON_COLOR = (140, 79, 29)


@dataclass
class PipelineOutput:
    session: SessionResult
    annotated_video_path: Optional[Path]
    annotation_note: Optional[str]  # e.g. "ffmpeg not found" so the UI can explain a missing video


def _noop_progress(stage: str, frac: float) -> None:
    pass


def run_pipeline(
    video_info: VideoInfo,
    bar_init_point: tuple[float, float],
    skill_path: Path,
    output_dir: Path,
    on_progress: ProgressFn = _noop_progress,
    target_shape: TargetShape = "plate",
) -> PipelineOutput:
    output_dir.mkdir(parents=True, exist_ok=True)
    skill = assessor.load_skill(skill_path)

    # Frames are streamed, never collected: a 30s portrait clip held in memory
    # at 720p is ~800 MB, over Streamlit Community Cloud's 690 MB minimum.
    # The clip is read in separate passes instead (decoding is cheap).

    # 1. Count frames and pick the side-choice samples.
    n_frames = count_frames(video_info)
    if n_frames == 0:
        raise RuntimeError("No frames could be read from this video.")
    sample_idxs = set(np.linspace(0, n_frames - 1, num=min(10, n_frames), dtype=int).tolist())
    samples = []
    for idx, _, frame in extract_frames(video_info):
        if idx in sample_idxs:
            samples.append(frame)
            if len(samples) == len(sample_idxs):
                break
    on_progress("Validating video", 1.0)

    # 2+3. Pose and bar tracking in one streaming pass.
    extractor = PoseExtractor()
    side = extractor.choose_side(samples)
    del samples

    poses = []
    bar_points = []
    bar_tracker = None
    for i, ts, frame in extract_frames(video_info):
        poses.append(extractor.extract(i, frame, side, ts))
        if bar_tracker is None:
            bar_tracker = BarTracker(frame, bar_init_point, target_shape)
            bar_points.append(bar_tracker.first_point)
        else:
            bar_points.append(bar_tracker.update(i, frame))
        if i % 5 == 0:
            on_progress("Pose tracking", i / max(n_frames - 1, 1))
    extractor.close()
    on_progress("Pose tracking", 1.0)
    on_progress("Tracking the bar", 1.0)

    # 4. Rep detection over the hip trajectory.
    hip_y_series = [p.points["hip"][1] if p else None for p in poses]
    shoulder_y_series = [p.points["shoulder"][1] if p else None for p in poses]
    reps = detect_reps(hip_y_series, video_info.fps)
    on_progress("Detecting repetitions", 1.0)

    # 5. Measurements + rule-based assessment, per rep.
    findings: list[Finding] = []
    for rep in reps:
        pose_at_bottom = poses[rep.bottom_frame]
        bar_at_bottom = bar_points[rep.bottom_frame] if rep.bottom_frame < len(bar_points) else None

        if pose_at_bottom is not None:
            depth_m = meas.measure_depth(pose_at_bottom.points.get("hip"), pose_at_bottom.points.get("knee"))
            back_m = meas.measure_back_angle(pose_at_bottom.points.get("shoulder"), pose_at_bottom.points.get("hip"))
        else:
            depth_m = meas.measure_depth(None, None)
            back_m = meas.measure_back_angle(None, None)

        if pose_at_bottom is not None and bar_at_bottom is not None and bar_at_bottom.x is not None:
            bar_m = meas.measure_bar_midfoot(
                (bar_at_bottom.x, bar_at_bottom.y),
                pose_at_bottom.points.get("ankle"),
                pose_at_bottom.points.get("foot_index"),
                bar_at_bottom.confidence,
            )
        else:
            bar_m = meas.measure_bar_midfoot(None, None, None, 0.0)

        hip_drive_m = meas.measure_hip_drive(hip_y_series, shoulder_y_series, rep.bottom_frame)

        findings.append(assessor.assess_depth(skill, rep.index, rep.bottom_time, depth_m))
        findings.append(assessor.assess_bar_midfoot(skill, rep.index, rep.bottom_time, bar_m))
        findings.append(assessor.assess_hip_drive(skill, rep.index, rep.bottom_time, hip_drive_m))
        findings.append(assessor.assess_back_angle(skill, rep.index, rep.bottom_time, back_m))
        for cannot_id in ("knee_tracking_frontal", "stance_width"):
            findings.append(assessor.assess_not_observable(skill, rep.index, cannot_id))

    on_progress("Assessing against the reference document", 1.0)

    # 6. Annotated video: skeleton + bar dot/trail + depth reference line.
    annotated_path, note = _write_annotated_video(poses, bar_points, video_info, output_dir)

    session = SessionResult(
        video_name=video_info.path.name,
        duration_seconds=video_info.duration_s,
        fps=video_info.fps,
        reps_detected=len(reps),
        findings=findings,
        skill_name=skill["skill"]["name"],
        skill_version=str(skill["skill"]["version"]),
    )

    return PipelineOutput(session=session, annotated_video_path=annotated_path, annotation_note=note)


def _write_annotated_video(poses, bar_points, video_info: VideoInfo, output_dir: Path):
    """Re-reads the clip frame by frame and draws on each one — see the
    memory note in run_pipeline()."""
    avi_path = output_dir / "annotated_raw.avi"
    writer = None

    bar_trail: list[tuple[int, int]] = []
    connections = [("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"), ("ankle", "foot_index"), ("ankle", "heel")]

    for i, _, frame in extract_frames(video_info):
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"MJPG"), video_info.fps, (w, h))
        canvas = frame
        pose = poses[i] if i < len(poses) else None
        bar = bar_points[i] if i < len(bar_points) else None

        if pose is not None:
            for a, b in connections:
                pa, pb = pose.points.get(a), pose.points.get(b)
                if pa and pb:
                    cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), SKELETON_COLOR, 2)
            for name, p in pose.points.items():
                cv2.circle(canvas, (int(p[0]), int(p[1])), 4, SKELETON_COLOR, -1)

        if bar is not None and bar.x is not None:
            color = BAR_COLOR if not bar.flagged else (80, 80, 200)
            bar_trail.append((int(bar.x), int(bar.y)))
            for j in range(1, len(bar_trail)):
                cv2.line(canvas, bar_trail[j - 1], bar_trail[j], color, 1)
            cv2.circle(canvas, (int(bar.x), int(bar.y)), 8, color, -1)

        writer.write(canvas)

    if writer is None:
        return None, "No frames to annotate."
    writer.release()

    mp4_path = output_dir / "annotated.mp4"
    ok = reencode_for_browser(avi_path, mp4_path)
    if ok:
        return mp4_path, None
    return avi_path, "Could not encode H.264 (install ffmpeg) — showing the raw .avi output, which may not play inline in your browser."
