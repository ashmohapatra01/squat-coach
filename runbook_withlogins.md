# Squat Coach AI — Runbook

How to install, run, and smoke-test the MVP from a fresh copy of this
project (e.g. unzipped from `squat-coach-package.zip`). Every command runs
from the project root — the folder containing this file.

## 1. Prerequisites

| Requirement | Why | Check |
|---|---|---|
| macOS (tested on Apple Silicon) | Only platform verified so far | — |
| Python 3.14 | Version the pinned dependencies were tested with | `python3 --version` |
| Internet access | Installing packages and downloading the pose model | — |
| ffmpeg (optional) | Faster, standard H.264 encode of the annotated video. Without it, the app falls back to OpenCV's H.264 writer, which works on macOS. | `ffmpeg -version` |

Install ffmpeg with `brew install ffmpeg` if you want it.

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/generate_credentials.py
```

- The pose model (~9 MB) is not shipped in the package. The app downloads
  it automatically into `models/` on the first analysis (adds a few seconds,
  needs internet access). To fetch it up front instead — e.g. for a machine
  that will run offline — run:
  ```bash
  mkdir -p models && curl -L -o models/pose_landmarker_full.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
  ```
- `generate_credentials.py` writes `credentials.yaml` (bcrypt hashes and a
  random cookie key) and prints the two demo passwords **once** — note them.
  They can't be read back from the file; to reset them, run
  `python scripts/generate_credentials.py --force`. Never ship or commit
  `credentials.yaml`.
- `.env` is optional. Put an `ANTHROPIC_API_KEY` in it to get the
  plain-language "Overall" summary; without a key you still get every
  finding.

## 3. Run

```bash
source .venv/bin/activate
streamlit run app/ui.py
```

Open http://localhost:8501 and log in with a demo account printed in
step 2: `athlete_demo` (athlete role) or `trainer_demo` (trainer role).
The login lasts 7 days via a cookie; **Logout** is in the sidebar. Stop the
server with `Ctrl+C`.

## 4. Smoke test (about 2 minutes)

Uses a public side-view squat clip, so no local video is needed.

1. Under **Video source**, choose **Paste a URL (testing)**.
2. Paste `https://www.youtube.com/shorts/TRvg083BrXY` and click **Fetch video**.
3. Expect: **"Looks good — 7.9s at 30fps, 608×1080."** and a caption
   "This is a portrait video; framing may cut off the bar or feet at the edges."
4. Leave **Plate** selected and click the center of the plate on the frame.
   A caption "Bar init point: (x, y)" appears.
5. Click **Analyze Squat**. Expect about 10 s of processing, then
   **"2 rep(s) detected"**, an annotated video that plays (skeleton plus bar
   dot and trail), and a per-rep findings table.

To test trimming, upload any clip over 30 s: it is rejected with the
**Trim clip** panel opened. Pick a 5–30 s range and click **Apply trim**.

## 5. Input limits

- MP4 or MOV, 5–30 s (longer clips can be trimmed in the app).
- Side view, whole body, feet and bar in frame, fixed camera, one lifter.
- Portrait video is allowed with a warning.
- Front-view clips are not supported: front-view criteria (knee tracking,
  stance width) always report "cannot assess", and the side-view
  measurements are not valid from the front.

## 6. Where things live

| Path | Contents |
|---|---|
| `app/ui.py` | Streamlit screens: upload/fetch/trim → processing → results |
| `app/pipeline.py` | Orchestrates pose → bar tracking → reps → assessment → annotated video |
| `app/pose.py` | MediaPipe Tasks pose landmarker (CPU) |
| `app/bar_tracker.py` | Click-to-init CSRT bar tracking; plate or bare-bar mode |
| `app/rep_detector.py` | Rep state machine over hip height |
| `app/auth.py`, `scripts/generate_credentials.py` | Login gate and username → role mapping; one-time credential setup |
| `app/trim.py`, `app/url_fetch.py` | In-app trimming; yt-dlp URL fetch (testing) |
| `skills/low_bar_squat_v1.yaml` | The assessment rules and citations — edit thresholds here |
| `$TMPDIR/squat_coach_sessions/<id>/` | Per-browser-session working files; folders older than 24 h are deleted automatically |

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `credentials.yaml not found` on startup | Run `python scripts/generate_credentials.py`. |
| Lost the demo passwords | `python scripts/generate_credentials.py --force` prints new ones. |
| `Pose model not found ... automatic download failed` | No internet access or the download was blocked; run the optional `curl` command from step 2 (the error message includes it). |
| Crash mentioning `DrishtiMetalHelper` / `Service is unavailable` | mediapipe 1.0.x is installed. `pip install "mediapipe>=0.10.30,<1.0"`. |
| `OpenCV build has no CSRT tracker available` | `pip install opencv-contrib-python` |
| Annotated video won't play | Install ffmpeg (`brew install ffmpeg`) and re-run the analysis. |
| Port 8501 already in use | Another Streamlit is running; stop it, or add `--server.port 8502`. |

## 8. Known issues and status

- **Hip-drive noise floor — not done.** "Hips lead the ascent" is decided by
  the frame difference between hip and shoulder starting to rise; a ±1-frame
  difference is within tracking noise but still produces meets / does-not-meet
  (`app/assessor.py`, `assess_hip_drive`). Planned: report ±1 frame as
  "cannot assess".
- **Bar-confidence re-check granularity — not done.** The plate-shape
  re-check runs only every 10th frame (`REVERIFY_EVERY_N_FRAMES = 10` in
  `app/bar_tracker.py`), so a rep's bar-over-midfoot confidence depends on
  whether its bottom frame lands on a check frame. Planned: re-check every
  frame near each rep's bottom.
- Bare-bar tracking mode never flags drift (fixed 0.75 confidence) and is
  untested on real bare-bar footage.
- Depth compares MediaPipe's hip joint center with the knee, not the hip
  crease, so it may read slightly shallow.
- Page numbers in `skills/low_bar_squat_v1.yaml` are placeholders (`[TBD]`).
