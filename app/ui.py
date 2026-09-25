"""Squat Coach AI — Streamlit MVP.

Three stages, driven by st.session_state.stage: "upload" -> "processing" -> "results".
Run with:  streamlit run app/ui.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

# `streamlit run app/ui.py` only puts this file's own directory on sys.path,
# not the project root — add it explicitly so `from app import ...` resolves
# no matter where you launch it from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from app import narrator, trim, video
from app.auth import require_login
from app.pipeline import run_pipeline
from app.styles import CSS
from app.url_fetch import download_video_from_url

try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    _HAS_CLICK_COMPONENT = True
except ImportError:
    _HAS_CLICK_COMPONENT = False

load_dotenv()

SKILL_PATH = Path(__file__).parent.parent / "skills" / "low_bar_squat_v1.yaml"
# One folder per browser session: a shared folder let concurrent sessions
# overwrite each other's annotated.mp4 and delete each other's files on reset.
WORKDIR_ROOT = Path(tempfile.gettempdir()) / "squat_coach_sessions"
STALE_WORKDIR_AGE_S = 24 * 60 * 60

st.set_page_config(page_title="Squat Coach AI", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# Login gate: halts this run (st.stop) until the user is authenticated.
require_login()



def _prune_stale_workdirs():
    """Sessions that are closed without "Analyze another video" never clean
    up after themselves — drop their folders once they're a day old."""
    if not WORKDIR_ROOT.exists():
        return
    cutoff = time.time() - STALE_WORKDIR_AGE_S
    for d in WORKDIR_ROOT.iterdir():
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def workdir() -> Path:
    return WORKDIR_ROOT / st.session_state.session_id


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    _prune_stale_workdirs()
if "stage" not in st.session_state:
    st.session_state.stage = "upload"
if "bar_point" not in st.session_state:
    st.session_state.bar_point = None
if "target_shape" not in st.session_state:
    st.session_state.target_shape = "plate"


def header():
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### Squat Coach AI")
    with col2:
        st.markdown(
            '<div class="sc-tier-badge" style="text-align:right">MVP &middot; single-video analysis</div>',
            unsafe_allow_html=True,
        )
    st.divider()


def reset():
    st.session_state.stage = "upload"
    st.session_state.bar_point = None
    st.session_state.fetched_video_path = None
    st.session_state.target_shape = "plate"
    st.session_state.trimmed = None
    if workdir().exists():
        shutil.rmtree(workdir(), ignore_errors=True)


# ---------------------------------------------------------------- UPLOAD --
def trim_step(src: Path) -> Path:
    """Optional trim before validation, so clips outside the duration limits
    can be cut down instead of rejected. Returns the path to use from here on."""
    probe = trim.probe_duration(src)
    if probe is None:
        return src  # unreadable — let validate_video() explain why

    trimmed = st.session_state.get("trimmed")
    if trimmed and trimmed["source"] != src:
        trimmed = st.session_state.trimmed = None

    duration = round(probe[1], 1)
    out_of_range = not (video.MIN_DURATION_S <= duration <= video.MAX_DURATION_S)

    with st.expander("Trim clip", expanded=out_of_range and trimmed is None):
        if out_of_range:
            st.caption(
                f"This clip is {duration:.1f}s — pick a {video.MIN_DURATION_S}–{video.MAX_DURATION_S}s window to analyze."
            )
        start, end = st.slider(
            "Keep this range (seconds)",
            0.0,
            duration,
            (0.0, min(duration, float(video.MAX_DURATION_S))),
            step=0.1,
            key=f"trim_range_{src.name}",
        )
        if st.button("Apply trim"):
            with st.spinner("Trimming..."):
                result = trim.trim_video(src, start, end, workdir())
            if result.ok:
                st.session_state.trimmed = {"source": src, "path": result.path, "note": result.reason}
                st.session_state.bar_point = None
                st.rerun()
            st.error(result.reason)

        if trimmed:
            st.caption(f"Using trimmed clip: {trimmed['note']}")
            if st.button("Use full clip"):
                st.session_state.trimmed = None
                st.session_state.bar_point = None
                st.rerun()

    return trimmed["path"] if trimmed else src


def stage_upload():
    header()
    left, right = st.columns([1, 1.4], gap="large")

    with left:
        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        st.markdown("#### Before you record")
        st.caption("Follow these so we can actually measure your reps — not just guess.")
        st.markdown(
            "- **Side view.** Camera perpendicular to the bar path.\n"
            "- **Full body, feet and bar visible.** Nothing cropped out of frame.\n"
            "- **Fixed camera.** Tripod or propped — no handheld panning.\n"
            "- **One lifter.** Others should stay out of frame."
        )
        st.markdown("**Format & limits**")
        st.markdown(
            f"MP4 or MOV &middot; {video.MIN_DURATION_S}–{video.MAX_DURATION_S}s &middot; "
            f"resized to ~{video.TARGET_HEIGHT}p for processing",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="sc-callout-orange" style="margin-top:12px">'
            "These are application limits for reliable tracking — not requirements from the "
            "reference document itself.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        source = st.radio("Video source", ["Upload a file", "Paste a URL (testing)"], horizontal=True)

        if source == "Upload a file":
            uploaded = st.file_uploader("Drag & drop your squat video", type=["mp4", "mov"])

            if uploaded is None:
                st.info("Upload a video to continue.")
                return

            workdir().mkdir(parents=True, exist_ok=True)
            video_path = workdir() / uploaded.name
            video_path.write_bytes(uploaded.getvalue())
        else:
            url = st.text_input("Video URL", placeholder="https://www.youtube.com/shorts/...")
            if st.button("Fetch video"):
                with st.spinner("Downloading video..."):
                    try:
                        st.session_state.fetched_video_path = download_video_from_url(url, workdir())
                        st.session_state.bar_point = None
                    except RuntimeError as e:
                        st.session_state.fetched_video_path = None
                        st.error(str(e))

            # Kept in session_state so it survives the rerun triggered by every widget interaction.
            video_path = st.session_state.get("fetched_video_path")
            if video_path is None or not video_path.exists():
                st.info("Paste a URL and fetch the video to continue.")
                return

        video_path = trim_step(video_path)

        result = video.validate_video(video_path)
        if not result.ok:
            st.error(f"This video won't work: {result.reason}")
            return

        st.session_state.video_info = result.info
        st.success(f"Looks good — {result.info.duration_s:.1f}s at {result.info.fps:.0f}fps, {result.info.width}×{result.info.height}.")
        if result.warning:
            st.caption(f"⚠️ {result.warning}")

        # Click-to-init the bar tracker on the first frame.
        first_frame = video.get_first_frame(result.info)

        if first_frame is None:
            st.error("Couldn't read the first frame.")
            return

        import cv2
        from PIL import Image

        rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        target_label = st.radio(
            "What's on the end of the bar?",
            ["Plate", "Bare bar (no plates)"],
            index=0 if st.session_state.target_shape == "plate" else 1,
            horizontal=True,
        )
        st.session_state.target_shape = "plate" if target_label == "Plate" else "bar"
        if st.session_state.target_shape == "plate":
            st.markdown("**Click the visible end of the barbell plate on this frame:**")
        else:
            st.markdown("**Click the end of the bare bar on this frame:**")
        if _HAS_CLICK_COMPONENT:
            # Keyed per file version so a fetched or trimmed clip doesn't inherit the previous clip's click.
            click_key = f"bar_click_{video_path.name}_{video_path.stat().st_mtime_ns}"
            coords = streamlit_image_coordinates(pil_img, key=click_key)
            if coords is not None:
                st.session_state.bar_point = (coords["x"], coords["y"])
        else:
            st.warning("`streamlit-image-coordinates` isn't installed — enter the plate's pixel coordinates manually.")
            st.image(pil_img)
            x = st.number_input("Plate x (px)", 0, pil_img.width, pil_img.width // 2)
            y = st.number_input("Plate y (px)", 0, pil_img.height, pil_img.height // 3)
            st.session_state.bar_point = (x, y)

        if st.session_state.bar_point:
            st.caption(f"Bar init point: {st.session_state.bar_point}")
            if st.button("Analyze Squat"):
                st.session_state.stage = "processing"
                st.rerun()


# ------------------------------------------------------------ PROCESSING --
def stage_processing():
    header()
    st.markdown("### Analyzing your squat")
    info = st.session_state.video_info
    st.caption(f"{info.path.name} · {info.duration_s:.0f}s · {video.TARGET_HEIGHT}p")

    status = st.status("Running the pipeline...", expanded=True)
    progress_bars = {}

    def on_progress(stage_name: str, frac: float):
        if stage_name not in progress_bars:
            status.write(f"**{stage_name}**")
            progress_bars[stage_name] = status.progress(0)
        progress_bars[stage_name].progress(min(frac, 1.0))

    try:
        output = run_pipeline(
            video_info=info,
            bar_init_point=st.session_state.bar_point,
            skill_path=SKILL_PATH,
            output_dir=workdir(),
            on_progress=on_progress,
            target_shape=st.session_state.target_shape,
        )
        status.update(label="Done", state="complete")
    except Exception as e:
        status.update(label="Failed", state="error")
        st.exception(e)
        if st.button("Back to upload"):
            reset()
            st.rerun()
        return

    summary_text = narrator.narrate(output.session.findings, info.path.name)
    output.session.overall_summary = summary_text

    st.session_state.pipeline_output = output
    st.session_state.stage = "results"
    st.rerun()


# ---------------------------------------------------------------- RESULTS --
def stage_results():
    header()
    output = st.session_state.pipeline_output
    session = output.session

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        if output.annotated_video_path and output.annotated_video_path.exists():
            st.video(str(output.annotated_video_path))
            if output.annotation_note:
                st.caption(output.annotation_note)
        else:
            st.warning("Annotated video could not be generated.")

        st.markdown(f"**{session.reps_detected} rep(s) detected**")

        st.markdown('<div class="sc-card">', unsafe_allow_html=True)
        st.markdown("**Overall**")
        st.write(session.overall_summary)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown(f"**Per-repetition findings** &nbsp;&middot;&nbsp; skill: `{session.skill_name} v{session.skill_version}`", unsafe_allow_html=True)

        rows = []
        for f in session.findings:
            rows.append(
                {
                    "Rep": f.rep,
                    "Criterion": f.criterion_title,
                    "Result": f.result.replace("_", " "),
                    "Time (s)": f"{f.timestamp_seconds:.2f}" if f.timestamp_seconds is not None else "—",
                    "Measurement": f"{f.measurement.value:.1f}{f.measurement.unit or ''}" if f.measurement.value is not None else "—",
                    "Confidence": f"{f.confidence:.2f}" if f.confidence is not None else "—",
                    "Source": f.source_section,
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        cannot = [f for f in session.findings if f.result == "cannot_assess"]
        if cannot:
            lines = "<br>".join(f"<b>Rep {f.rep} · {f.criterion_title}:</b> {f.reason}" for f in cannot[:4])
            st.markdown(
                f'<div class="sc-callout-orange"><b>Cannot-assess isn\'t a bug.</b> '
                f"{lines}</div>",
                unsafe_allow_html=True,
            )

        with st.expander("Raw findings (JSON)"):
            st.json([f.model_dump() for f in session.findings])

    st.divider()
    if st.button("Analyze another video"):
        reset()
        st.rerun()


# --------------------------------------------------------------- ROUTER --
if st.session_state.stage == "upload":
    stage_upload()
elif st.session_state.stage == "processing":
    stage_processing()
elif st.session_state.stage == "results":
    stage_results()
