"""Optional plain-language narration.

Only ever receives the already-decided findings (as JSON) — never raw video
or frames. It cannot change a measurement or a met/not-met/cannot-assess
verdict; those are fixed by app/assessor.py before this runs. If no API key
is configured, the app falls back to a simple templated summary with no
LLM call at all.
"""
from __future__ import annotations

import os
from typing import Optional

from app.schemas import Finding


def _templated_summary(findings: list[Finding]) -> str:
    met = sum(1 for f in findings if f.result == "meets")
    not_met = sum(1 for f in findings if f.result == "does_not_meet")
    cannot = sum(1 for f in findings if f.result == "cannot_assess")
    return (
        f"{met} of {met + not_met} scoreable checks met across your reps "
        f"({cannot} marked cannot-assess — see the findings table for why)."
    )


def narrate(findings: list[Finding], video_name: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return _templated_summary(findings)

    try:
        import anthropic
    except ImportError:
        return _templated_summary(findings)

    findings_json = [f.model_dump() for f in findings]
    system_prompt = (
        "You are writing a short, encouraging summary of squat-form findings for a lifter. "
        "You are given a list of already-decided findings (result, measurement, feedback). "
        "Do NOT change, add, or contradict any result or measurement — only summarize and "
        "prioritize what's already there in plain language, 3-4 sentences, no new claims."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Video: {video_name}\nFindings: {findings_json}"}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "".join(text_blocks).strip() or _templated_summary(findings)
    except Exception:
        # Network/auth/anything else: never let narration failure break the app.
        return _templated_summary(findings)
