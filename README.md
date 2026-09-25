# Squat Coach AI — MVP

A consumer app that takes a side-view squat video, tracks the lifter and
barbell, and assesses each rep against a reference document via an
inspectable skill file — not generic AI squat advice.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p models && curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
cp .env.example .env             # optional — add ANTHROPIC_API_KEY for narration
streamlit run app/ui.py
```

You also need **ffmpeg** on your PATH for the final annotated-video
re-encode (`brew install ffmpeg` / `apt install ffmpeg`). The app will
still run and produce findings without it — it just falls back to the raw
OpenCV `.avi` output, which most browsers won't play inline.

## What's real vs. stubbed in this pass

- **Real:** video validation, MediaPipe pose extraction, click-to-init bar
  tracking (OpenCV CSRT), rep segmentation (state machine over smoothed hip
  trajectory), rule-based assessment against `skills/low_bar_squat_v1.yaml`,
  annotated video rendering (skeleton + bar path overlay).
- **Optional:** the plain-language narration step calls Claude if
  `ANTHROPIC_API_KEY` is set. Without a key, you still get the full
  structured findings table — just no prose summary on top. The narrator
  never rewrites a measurement or a met/not-met/cannot-assess verdict; it
  only phrases what the rule engine already decided.
- **Known limitation:** bar re-verification (confirming the tracker hasn't
  drifted onto something else) is a simple brightness/circularity check on
  the tracked region, not a trained detector. Treat low-confidence bar
  frames in the output as "flagged," not "wrong."

## Editing the design

Colors, fonts, and layout live in `app/styles.py` (a single CSS block
injected into the page) — change hex values or the Google Fonts import
there and it applies everywhere. Screen structure (upload → processing →
results) is in `app/ui.py`.

## Editing the assessment rules

`skills/low_bar_squat_v1.yaml` is the whole rule set. Each criterion has a
`source` citation, an `assessability` (conditional / proxy_only / cannot),
and the measurement it's checked against. Edit a threshold or a citation
there and rerun — no code changes needed. This is the file to show
changing live during a walkthrough.

**Page numbers in the skill file are placeholders** (`source_pages: [TBD]`
or a figure reference) where I could confirm the section/figure from the
document text but not the exact printed page number in your actual PDF —
confirm and fill these in against your copy before treating them as final
citations.
